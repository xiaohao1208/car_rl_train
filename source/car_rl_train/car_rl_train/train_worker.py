"""YAML 与命令行共用的 split-fusion PPO 训练入口。"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

from car_rl_train.workflow import ROOT
from car_rl_train.assets import load_and_validate_asset_manifest
TRAINING_DEFINITION_VERSION = 44
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

from car_rl_train.workflow import (
    RUN_STATUS_NAME,
    apply_object_config,
    atomic_write_json,
    load_workflow_config,
    portable_project_value,
    project_relative_path,
    resolve_checkpoint,
)
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
known, _ = pre_parser.parse_known_args()
workflow = load_workflow_config(known.config)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(parents=[pre_parser])
parser.add_argument("--task", default=workflow.get("task", "Car-LocalNavigation-v0"))
parser.add_argument("--seed", type=int, default=workflow.get("seed", 42))
parser.add_argument("--num_envs", type=int, default=workflow.get("num_envs", 256))
parser.add_argument("--max_iterations", type=int, default=workflow.get("max_iterations", 8000))
parser.add_argument("--run_name", default=workflow.get("run_name"))
parser.add_argument("--checkpoint", type=str, default=workflow.get("checkpoint"))
parser.add_argument("--calibration", default=workflow.get("calibration"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=bool(workflow.get("headless", True)))
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import car_rl_train  # noqa: E402,F401
from car_rl_train.contract.schema import (  # noqa: E402
    ACTION_SIZE,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_SIZE,
    ONNX_OPSET,
)
from car_rl_train.models import CarOnPolicyRunner  # noqa: E402
from car_rl_train.runtime import configure_environment  # noqa: E402
from car_rl_train.workflow import checkpoint_metadata  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _timestamp() -> str:
    """返回带时区的运行审计时间。"""

    return datetime.now().astimezone().isoformat()


def _curriculum_environment_signature(value: dict) -> dict:
    """提取影响任务分布的课程字段，排除纯晋级统计门槛。"""

    return {
        "geometry": value.get("geometry"),
        "stages": [
            {
                key: item[key]
                for key in (
                    "stage",
                    "path_length_m",
                    "static_obstacles",
                    "dynamic_obstacles",
                    "max_curve_offset_m",
                    "goal_yaw_offset_deg",
                    "randomization_scale",
                )
            }
            for item in value.get("stages", [])
        ],
        "replay_previous_probability": value.get(
            "replay_previous_probability"
        ),
        "randomization": value.get("randomization"),
    }


def _latest_local_checkpoint(log_dir: Path) -> Path | None:
    """返回当前运行目录中编号最大的 checkpoint。"""

    candidates = []
    for path in log_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))[1]


def main() -> Path:
    """解析配置、记录可复现元数据并启动唯一生产 runner。"""

    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    asset_manifest = load_and_validate_asset_manifest()
    resume_checkpoint = resolve_checkpoint(args.checkpoint) if args.checkpoint else None
    resume_metadata = None
    if resume_checkpoint:
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resume_metadata = checkpoint_metadata(resume_payload, resume_checkpoint)
        if resume_metadata is None:
            raise ValueError("checkpoint缺少v44生产元数据，禁止恢复旧训练")
        if (
            resume_metadata.get("source", {}).get("source_fingerprint_sha256")
            != asset_manifest["source_fingerprint_sha256"]
        ):
            raise ValueError("checkpoint 的小车描述指纹与当前资产不一致")
        policy_meta = (resume_metadata or {}).get("policy", {}).get("kwargs", {})
        algorithm_meta = (resume_metadata or {}).get("algorithm", {}).get("kwargs", {})
        curriculum_meta = (resume_metadata or {}).get("curriculum_config", {})
        saved_stage = int(
            (resume_metadata or {})
            .get("training_diagnostics", {})
            .get("curriculum", {})
            .get("stage", -1)
        )
        resume_definition = resume_metadata.get("training_definition_version")
        compatible = (
            resume_definition == TRAINING_DEFINITION_VERSION
            and policy_meta.get("noise_std_type") == "bounded_logit"
            and float(policy_meta.get("min_noise_std", -1.0)) == 0.04
            and float(policy_meta.get("max_noise_std", -1.0)) == 0.10
            and algorithm_meta.get("schedule") == "fixed"
            and float(algorithm_meta.get("entropy_coef", -1.0)) == 0.0002
            and float(algorithm_meta.get("min_learning_rate", -1.0)) == 1.0e-5
            and float(algorithm_meta.get("lr_decay_half_life", -1.0)) == 50.0
            and algorithm_meta.get("symmetry_contract") == "car_mirror_v1"
            and curriculum_meta.get("schema_version") == 6
        )
        if not compatible:
            raise ValueError(
                "checkpoint来自旧训练定义，禁止恢复；v44必须从头训练"
            )
    policy_name = "split_fusion"
    algorithm_name = "ppo"
    device = args.device or workflow.get("device", "cuda:0")
    cuda_available = torch.cuda.is_available()
    if str(device).startswith("cuda") and not cuda_available:
        raise RuntimeError(
            f"CUDA device requested ({device}) but torch.cuda.is_available() is false"
        )
    backend = {
        "simulation_device": str(device),
        "policy_device": str(device),
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    print("execution_backend=" + json.dumps(backend, ensure_ascii=False), flush=True)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    curriculum_config, calibration, physics_calibration = configure_environment(
        env_cfg,
        workflow,
        training=True,
        calibration=args.calibration,
        checkpoint_metadata=resume_metadata,
    )
    if resume_metadata:
        checkpoint_curriculum = resume_metadata.get("curriculum_config")
        if checkpoint_curriculum != curriculum_config:
            saved_stage = int(
                resume_metadata.get("training_diagnostics", {})
                .get("curriculum", {})
                .get("stage", -1)
            )
            threshold_only_change = (
                resume_metadata.get("training_definition_version")
                == TRAINING_DEFINITION_VERSION
                and _curriculum_environment_signature(checkpoint_curriculum)
                == _curriculum_environment_signature(curriculum_config)
            )
            if not threshold_only_change:
                raise ValueError(
                    "checkpoint curriculum differs from the active curriculum config"
                )
            migration = "v44_reciprocal_dynamic_front_brake"
            print(f"resume_curriculum_migration={migration}", flush=True)
        saved_curriculum = (
            resume_metadata.get("training_diagnostics", {}).get("curriculum", {})
        )
        saved_stage = int(saved_curriculum.get("stage", -1))
        if saved_stage not in range(len(curriculum_config["stages"])):
            raise ValueError("checkpoint lacks a valid saved curriculum stage")
        env_cfg.commands.navigation.curriculum_stage = saved_stage
        saved_stage_episode_total = int(
            saved_curriculum.get("stage_episode_total", 0)
        )
        if saved_stage_episode_total < 0:
            raise ValueError("checkpoint has a negative stage episode total")
        env_cfg.commands.navigation.resume_stage_episode_total = (
            saved_stage_episode_total
        )
        print(f"resume_curriculum_stage={saved_stage}", flush=True)
        print(
            f"resume_stage_episode_total={saved_stage_episode_total}",
            flush=True,
        )
    print(
        "physics_calibration="
        + json.dumps(physics_calibration, sort_keys=True, ensure_ascii=False)
    )
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = device
    agent_cfg.max_iterations = args.max_iterations
    policy_values = (
        resume_metadata.get("policy", {}).get("kwargs", {})
        if resume_metadata
        else workflow.get("policy", {}).get("kwargs", {})
    )
    apply_object_config(agent_cfg.policy, policy_values, "policy")
    algorithm_values = (
        resume_metadata.get("algorithm", {}).get("kwargs", {})
        if resume_metadata
        else workflow.get("algorithm", {}).get("kwargs", {})
    )
    apply_object_config(agent_cfg.algorithm, algorithm_values, "algorithm")
    runner_options = workflow.get("runner", {})
    apply_object_config(agent_cfg, runner_options, "runner")
    suffix = args.run_name or f"{policy_name}_{algorithm_name}_seed{args.seed}"
    log_dir = (
        ROOT
        / "logs"
        / "rsl_rl"
        / agent_cfg.experiment_name
        / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{suffix}"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    status_path = log_dir / RUN_STATUS_NAME
    run_status = {
        "schema_version": 1,
        "status": "running",
        "started_at": _timestamp(),
        "finished_at": None,
        "run_name": suffix,
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "requested_iterations": int(args.max_iterations),
        "resume_checkpoint": (
            project_relative_path(resume_checkpoint)
            if resume_checkpoint is not None else None
        ),
        "initial_curriculum_stage": (
            int(saved_stage) if resume_metadata else int(curriculum_config["initial_stage"])
        ),
        "initial_stage_episode_total": (
            int(saved_stage_episode_total) if resume_metadata else 0
        ),
        "last_completed_iteration": None,
        "last_checkpoint": None,
        "final_checkpoint": None,
        "backend": backend,
        "failure": None,
        "training_eligible": False,
        "eligibility_failures": ["training has not completed"],
        "eligible_checkpoints": [],
    }
    atomic_write_json(status_path, run_status)
    checkpoint_info = {
        "schema_version": 2,
        "training_definition_version": TRAINING_DEFINITION_VERSION,
        "policy": {"name": policy_name, "kwargs": policy_values},
        "algorithm": {"name": algorithm_name, "kwargs": algorithm_values},
        "seed": args.seed,
        "task": args.task,
        "contracts": {
            "observation_version": OBSERVATION_CONTRACT_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "onnx_opset": ONNX_OPSET,
        },
        "source": asset_manifest,
        "calibration": calibration,
        "physics_calibration": physics_calibration,
        "curriculum_config": curriculum_config,
        "environment": portable_project_value(env_cfg.to_dict()),
        "workflow_config": portable_project_value(workflow),
    }
    resolved = {
        "schema_version": 2,
        "task": args.task,
        "model": policy_name,
        "algorithm": algorithm_name,
        "seed": args.seed,
        "environment": portable_project_value(env_cfg.to_dict()),
        "runner": agent_cfg.to_dict(),
        "source": asset_manifest,
        "checkpoint_metadata": checkpoint_info,
        "calibration": calibration,
        "physics_calibration": physics_calibration,
        "curriculum_config": curriculum_config,
    }
    atomic_write_json(log_dir / "resolved_run_config.json", resolved)
    previous_signal_handlers = {}

    def _raise_interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, _raise_interrupted)
    env = None
    runner = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = CarOnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=device,
            checkpoint_metadata=checkpoint_info,
        )
        if resume_checkpoint:
            runner.load(str(resume_checkpoint))
        runner.add_git_repo_to_log(__file__)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    except BaseException as error:
        checkpoint = _latest_local_checkpoint(log_dir)
        run_status.update(
            {
                "status": "failed",
                "finished_at": _timestamp(),
                "last_completed_iteration": (
                    int(runner.current_learning_iteration)
                    if runner is not None
                    else None
                ),
                "last_checkpoint": checkpoint.name if checkpoint else None,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        atomic_write_json(status_path, run_status)
        raise
    finally:
        if env is not None:
            active_error = sys.exc_info()[1]
            try:
                env.close()
            except BaseException as cleanup_error:
                if active_error is None:
                    checkpoint = _latest_local_checkpoint(log_dir)
                    run_status.update(
                        {
                            "status": "failed",
                            "finished_at": _timestamp(),
                            "last_completed_iteration": (
                                int(runner.current_learning_iteration)
                                if runner is not None
                                else None
                            ),
                            "last_checkpoint": (
                                checkpoint.name if checkpoint else None
                            ),
                            "failure": {
                                "type": type(cleanup_error).__name__,
                                "message": f"environment cleanup failed: {cleanup_error}",
                            },
                        }
                    )
                    atomic_write_json(status_path, run_status)
                    raise
                print("environment cleanup failed after training error:", file=sys.stderr)
                traceback.print_exc()
        for signal_number, handler in previous_signal_handlers.items():
            signal.signal(signal_number, handler)

    final_iteration = int(runner.current_learning_iteration)
    final_checkpoint = log_dir / f"model_{final_iteration}.pt"
    if not final_checkpoint.is_file():
        error = RuntimeError(
            f"training returned without final checkpoint: {final_checkpoint}"
        )
        run_status.update(
            {
                "status": "failed",
                "finished_at": _timestamp(),
                "last_completed_iteration": final_iteration,
                "last_checkpoint": (
                    _latest_local_checkpoint(log_dir).name
                    if _latest_local_checkpoint(log_dir)
                    else None
                ),
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        atomic_write_json(status_path, run_status)
        raise error
    eligible = []
    final_stage_required = len(curriculum_config["stages"]) - 1
    final_stage_cfg = curriculum_config["stages"][final_stage_required]
    required_success = float(final_stage_cfg["success_rate"])
    maximum_collision = float(final_stage_cfg["max_collision_rate"])
    minimum_stage_episodes = int(final_stage_cfg["min_stage_episodes"])
    final_diagnostics = None

    def checkpoint_iteration(path: Path) -> int:
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    checkpoints = sorted(log_dir.glob("model_*.pt"), key=checkpoint_iteration)
    for checkpoint in reversed(checkpoints):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = checkpoint_metadata(payload, checkpoint) or {}
        diagnostics = metadata.get("training_diagnostics", {})
        curriculum_diagnostics = diagnostics.get("curriculum", {})
        if checkpoint == final_checkpoint:
            final_diagnostics = diagnostics
        checkpoint_stage = int(curriculum_diagnostics.get("stage", -1))
        checkpoint_success = float(
            curriculum_diagnostics.get("window_success_rate", -1.0)
        )
        checkpoint_collision = float(
            curriculum_diagnostics.get("window_collision_rate", 1.0)
        )
        checkpoint_stage_episodes = int(
            curriculum_diagnostics.get("stage_episode_total", 0)
        )
        if (
            checkpoint_stage == final_stage_required
            and checkpoint_stage_episodes >= minimum_stage_episodes
            and checkpoint_success >= required_success
            and checkpoint_collision <= maximum_collision
        ):
            eligible.append(checkpoint)
            if len(eligible) == 5:
                break
    eligible.reverse()
    final_stage = int(
        ((final_diagnostics or {}).get("curriculum") or {}).get("stage", -1)
    )
    final_curriculum = (final_diagnostics or {}).get("curriculum") or {}
    eligibility_failures = []
    final_stage_episodes = int(final_curriculum.get("stage_episode_total", 0))
    if final_stage != final_stage_required:
        eligibility_failures.append(
            f"final curriculum stage is {final_stage}, required {final_stage_required}"
        )
    if len(eligible) < 5:
        eligibility_failures.append(
            f"only {len(eligible)} qualified final-stage checkpoints, required 5"
        )
    if final_stage == final_stage_required and final_stage_episodes < minimum_stage_episodes:
        eligibility_failures.append(
            f"final-stage episodes are {final_stage_episodes}, required >= {minimum_stage_episodes}"
        )
    run_status.update(
        {
            "status": "completed",
            "finished_at": _timestamp(),
            "last_completed_iteration": final_iteration,
            "last_checkpoint": final_checkpoint.name,
            "final_checkpoint": final_checkpoint.name,
            "failure": None,
            "training_eligible": not eligibility_failures,
            "eligibility_failures": eligibility_failures,
            "eligible_checkpoints": [path.name for path in eligible],
            "final_training_diagnostics": final_diagnostics,
        }
    )
    atomic_write_json(status_path, run_status)
    print(f"log_dir={log_dir}", flush=True)
    print(f"run_status={status_path}", flush=True)
    print(f"final_checkpoint={final_checkpoint}", flush=True)
    return final_checkpoint


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise SystemExit(1)
    else:
        simulation_app.close()
