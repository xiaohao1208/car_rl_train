"""[必选] 在固定未见地图上评估候选 checkpoint，并选择正式模型。"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source" / "car_rl_train"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from car_rl_train.workflow import (
    load_workflow_config,
    project_relative_path,
    resolve_checkpoint,
    resolve_path,
)


SCRIPT = Path(__file__).resolve()
RESULT_FILES = (
    "episodes.jsonl",
    "evaluation.json",
    "simulation_observations.npz",
    "simulation_observations.manifest.json",
)
def suite_definition(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    suites = workflow.get("suites")
    if workflow.get("schema_version") != 2 or not isinstance(suites, dict):
        raise ValueError("evaluation config must use schema_version=2 with suites")
    if name not in suites or not isinstance(suites[name], dict):
        raise ValueError(f"unknown evaluation suite {name!r}; choices={sorted(suites)}")
    return suites[name]


def navigation_gate_failures(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> list[str]:
    """返回 checkpoint 在仿真导航门禁上的全部失败原因。"""

    failures = []
    for configured_name, threshold in gates.items():
        if configured_name.endswith("_min"):
            name, operator = configured_name[:-4], ">="
        elif configured_name.endswith("_max"):
            name, operator = configured_name[:-4], "<="
        else:
            raise ValueError(f"gate must end in _min or _max: {configured_name}")
        if name not in metrics:
            failures.append(f"{name}=missing")
            continue
        value = float(metrics[name])
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value} requires {operator}{threshold}")
    return failures


def candidate_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    """与生产评估模块一致的安全优先候选排序键。"""

    return (
        float(metrics["collision_rate"]),
        -float(metrics["success_rate"]),
        float(metrics["p95_cross_track_error_m"]),
        float(metrics["mean_action_delta"]),
    )


def build_common_parser() -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    """读取工作流配置，并构造调度器和worker共用参数。"""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "evaluate.yaml"
    )
    pre_parser.add_argument("--suite", choices=("core", "dynamic", "stress", "all"))
    known, _ = pre_parser.parse_known_args()
    workflow = load_workflow_config(known.config)
    selected_suite = known.suite or workflow.get("suite", "core")
    defaults = suite_definition(
        workflow, "core" if selected_suite == "all" else selected_suite
    )
    seed_range = defaults["seed_range"]
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument(
        "--campaign",
        default=workflow.get("campaign"),
        help="06_train.py 生成的已完成 campaign JSON",
    )
    parser.add_argument(
        "--tail-checkpoints",
        type=int,
        default=None,
        help="checkpoints=latest时评估同一训练目录末段N个checkpoint",
    )
    parser.add_argument("--num_envs", type=int, default=workflow.get("num_envs", 16))
    parser.add_argument("--episodes", type=int, default=int(defaults["episodes"]))
    parser.add_argument(
        "--seed-start", type=int, default=int(seed_range[0])
    )
    parser.add_argument(
        "--seed-end", type=int, default=int(seed_range[1])
    )
    parser.add_argument(
        "--output-root", default=workflow.get("output_root", "evaluations")
    )
    parser.add_argument("--calibration", default=workflow.get("calibration"))
    parser.add_argument(
        "--checkpoint-timeout-sec",
        type=int,
        default=workflow.get("checkpoint_timeout_sec", 600),
        help="每个checkpoint独立子进程的最长运行时间",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(suite=selected_suite)
    return parser, workflow


def checkpoint_paths(
    checkpoint_values: list[str] | str | None,
    configured_values: list[str] | str,
    tail_checkpoints: int | None,
) -> list[Path]:
    """展开latest、路径列表和glob，同时稳定去重。"""

    values = checkpoint_values or configured_values
    if not isinstance(values, list):
        values = [values]
    if tail_checkpoints is not None:
        if tail_checkpoints <= 0:
            raise ValueError("tail_checkpoints must be positive")
        if len(values) != 1 or str(values[0]).lower() != "latest":
            raise ValueError("tail_checkpoints只允许与唯一的latest选择一起使用")
        latest = resolve_checkpoint("latest")

        def iteration(path: Path) -> int:
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            return int(match.group(1)) if match else -1

        matches = sorted(latest.parent.glob("model_*.pt"), key=iteration)
        selected = matches[-tail_checkpoints:]
        if len(selected) < tail_checkpoints:
            raise FileNotFoundError(
                f"训练目录只有{len(selected)}个checkpoint，"
                f"不足tail_checkpoints={tail_checkpoints}"
            )
        return [path.resolve() for path in selected]
    result: list[Path] = []
    for value in values:
        if str(value).lower() == "latest":
            matches = [resolve_checkpoint("latest")]
        elif any(token in str(value) for token in "*?["):
            pattern = str(resolve_path(value))
            matches = [Path(item).resolve() for item in glob.glob(pattern)]
        else:
            matches = [resolve_checkpoint(value)]
        for path in matches:
            if path not in result:
                result.append(path)
    if not result:
        raise FileNotFoundError("evaluation checkpoint selection is empty")
    return result


def campaign_checkpoint_paths(value: str | Path) -> list[Path]:
    """严格解析已完成训练 campaign 中固定的候选列表。"""

    path = resolve_path(value, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == 1 and "campaign" in payload:
        path = resolve_path(payload["campaign"], must_exist=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in (1, 2)
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"campaign is not completed: {path}")
    values = payload.get("checkpoints")
    if not isinstance(values, list) or not values:
        raise ValueError(f"campaign checkpoint selection is empty: {path}")
    result = checkpoint_paths(values, values, None)
    expected_seeds = payload.get("seeds")
    runs = payload.get("runs")
    if (
        not isinstance(expected_seeds, list)
        or not isinstance(runs, list)
        or sorted(int(item["seed"]) for item in runs)
        != sorted(int(seed) for seed in expected_seeds)
    ):
        raise ValueError(f"campaign run/seed mapping is invalid: {path}")
    if payload.get("schema_version") == 2:
        if any(
            run.get("status") != "completed"
            or run.get("selected_checkpoint") not in values
            or not isinstance(run.get("segments"), list)
            or not run["segments"]
            for run in runs
        ):
            raise ValueError(f"campaign v2 segments or selections are invalid: {path}")
    return result


def candidate_label(checkpoint: Path, suite: str = "core") -> str:
    """生成与历史评估目录兼容的稳定候选名称。"""

    return f"{checkpoint.parent.name}__{checkpoint.stem}__{suite}"


def apply_suite_profile(env_cfg: Any, suite: dict[str, Any]) -> None:
    """用评估 suite 覆盖最终阶段任务分布，不改变训练 checkpoint。"""

    profile = suite.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("evaluation suite profile must be a mapping")
    path_range = profile["path_length_m"]
    static_range = profile["static_obstacles"]
    dynamic_range = profile["dynamic_obstacles"]
    target = env_cfg.commands.navigation.stage_profiles[-1]
    target.update(
        {
            "min_path_length_m": float(path_range[0]),
            "max_path_length_m": float(path_range[1]),
            "min_obstacles": int(static_range[0]),
            "max_obstacles": int(static_range[1]),
            "min_dynamic_obstacles": int(dynamic_range[0]),
            "max_dynamic_obstacles": int(dynamic_range[1]),
            "max_curve_offset_m": float(
                profile.get("max_curve_offset_m", target["max_curve_offset_m"])
            ),
            "goal_yaw_offset_rad": float(profile["goal_yaw_offset_deg"])
            * math.pi
            / 180.0,
            "randomization_scale": float(profile["randomization_scale"]),
        }
    )


def validate_candidate_output(
    checkpoint: Path,
    destination: Path,
    episodes: int,
    seed_start: int,
    seed_end: int,
    suite: str,
) -> dict[str, Any]:
    """严格检查一个候选的四个结果文件，并返回排名记录。"""

    missing = [name for name in RESULT_FILES if not (destination / name).is_file()]
    if missing:
        raise ValueError(f"候选结果缺少文件: {', '.join(missing)}")
    lines = [
        line
        for line in (destination / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(lines) != episodes:
        raise ValueError(f"逐回合结果数量无效: {len(lines)} != {episodes}")
    records = [json.loads(line) for line in lines]
    seeds = [int(record["seed"]) for record in records]
    expected_seeds = list(range(seed_start, seed_end + 1))
    if seeds != expected_seeds or episodes != len(expected_seeds):
        raise ValueError("逐回合测试种子不完整或顺序不正确")

    metrics = json.loads((destination / "evaluation.json").read_text(encoding="utf-8"))
    if metrics.get("schema_version") != 4 or int(metrics.get("episodes", 0)) != episodes:
        raise ValueError("评估汇总schema或回合数无效")
    if metrics.get("evaluation_suite") != suite:
        raise ValueError("评估suite与请求不一致")
    if metrics.get("test_seed_range") != [seed_start, seed_end]:
        raise ValueError("评估汇总种子范围无效")
    candidate_sort_key(metrics)

    manifest = json.loads(
        (destination / "simulation_observations.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(destination / "simulation_observations.npz") as payload:
        observations = payload["observation"]
    expected_shape = [int(observations.shape[0]), 86]
    checks = (
        manifest.get("schema_version") == 1,
        manifest.get("source") == "simulation",
        manifest.get("production_eligible") is False,
        manifest.get("shape") == expected_shape,
        int(manifest.get("count", -1)) == expected_shape[0],
        observations.ndim == 2,
        observations.shape[1] == 86,
        bool(np.all(np.isfinite(observations))),
    )
    if not all(checks):
        raise ValueError("仿真观测或manifest不完整")
    return {
        "checkpoint": project_relative_path(checkpoint),
        "output": project_relative_path(destination),
        "metrics": metrics,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    """在同一目录先落盘pending文件，再原子替换JSON。"""

    pending = path.with_name(f".{path.name}.pending.{os.getpid()}")
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(pending, path)


def quarantine_incomplete(destination: Path) -> Path:
    """可恢复地隔离旧版或中断留下的不完整候选目录。"""

    suffix = time.strftime("%Y%m%d_%H%M%S")
    quarantine = destination.with_name(f"{destination.name}.incomplete.{suffix}")
    counter = 1
    while quarantine.exists():
        quarantine = destination.with_name(
            f"{destination.name}.incomplete.{suffix}.{counter}"
        )
        counter += 1
    os.replace(destination, quarantine)
    return quarantine


def worker_command(args: argparse.Namespace, checkpoint: Path, output_root: Path) -> list[str]:
    """把调度器的有效配置完整传给单checkpoint worker。"""

    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(args.config.resolve()),
        "--suite",
        str(args.suite),
        "--worker",
        "--worker-checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--seed-start",
        str(args.seed_start),
        "--seed-end",
        str(args.seed_end),
        "--output-root",
        str(output_root),
        "--device",
        str(args.device),
    ]
    if args.calibration is not None:
        command.extend(["--calibration", str(args.calibration)])
    if args.headless:
        command.append("--headless")
    return command


def run_worker(command: list[str], timeout_sec: int, checkpoint: Path) -> None:
    """运行独立进程；超时或中断时终止整个Isaac Sim进程组。"""

    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise TimeoutError(
            f"checkpoint评估超过{timeout_sec}秒: {checkpoint}"
        ) from error
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"checkpoint评估子进程失败，退出码={return_code}: {checkpoint}")
    print(
        f"[scheduler] 子进程完成 checkpoint={checkpoint.name} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


def scheduler_main(args: argparse.Namespace, workflow: dict[str, Any]) -> None:
    """断点续跑全部候选，每个候选使用全新Isaac Sim进程。"""

    if args.suite == "all":
        root = resolve_path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for name in ("core", "dynamic", "stress"):
            definition = suite_definition(workflow, name)
            child = copy.copy(args)
            child.suite = name
            child.episodes = int(definition["episodes"])
            child.seed_start = int(definition["seed_range"][0])
            child.seed_end = int(definition["seed_range"][1])
            child.output_root = str(root / name)
            scheduler_main(child, workflow)
            summaries[name] = json.loads(
                (root / name / "best_checkpoint.json").read_text(encoding="utf-8")
            )
        rankings = {
            name: json.loads(
                (root / name / "candidate_ranking.json").read_text(encoding="utf-8")
            )
            for name in ("core", "dynamic", "stress")
        }
        indexed = {
            name: {item["checkpoint"]: item for item in ranking}
            for name, ranking in rankings.items()
        }
        common = set(indexed["core"]) & set(indexed["dynamic"])
        qualified = []
        for checkpoint in common:
            core_metrics = indexed["core"][checkpoint]["metrics"]
            dynamic_metrics = indexed["dynamic"][checkpoint]["metrics"]
            if not navigation_gate_failures(
                core_metrics, suite_definition(workflow, "core").get("gates", {})
            ) and not navigation_gate_failures(
                dynamic_metrics,
                suite_definition(workflow, "dynamic").get("gates", {}),
            ):
                qualified.append((checkpoint, core_metrics, dynamic_metrics))
        qualified.sort(
            key=lambda item: (
                float(item[1]["collision_rate"]),
                float(item[2]["collision_rate"]),
                -0.5
                * (float(item[1]["success_rate"]) + float(item[2]["success_rate"])),
            )
        )
        selected_checkpoint = qualified[0][0] if qualified else None
        suite_metrics = (
            {
                "core": indexed["core"][selected_checkpoint]["metrics"],
                "dynamic": indexed["dynamic"][selected_checkpoint]["metrics"],
                "stress": indexed["stress"].get(selected_checkpoint, {}).get("metrics"),
            }
            if selected_checkpoint
            else {}
        )
        overall = {
            "schema_version": 2,
            "checkpoint": selected_checkpoint,
            "suites": summaries,
            "suite_metrics": suite_metrics,
            "navigation_gates_passed": selected_checkpoint is not None,
            "stress_is_informational": True,
        }
        atomic_write_json(root / "suite_summary.json", overall)
        atomic_write_json(
            ROOT / "evaluations" / "latest_evaluation.json",
            {
                "schema_version": 1,
                "evaluation": project_relative_path(root / "suite_summary.json"),
                "checkpoint": selected_checkpoint,
            },
        )
        print(json.dumps(overall, indent=2), flush=True)
        return
    definition = suite_definition(workflow, args.suite)
    gates = definition.get("gates", {})
    if args.checkpoint_timeout_sec <= 0:
        raise ValueError("checkpoint_timeout_sec must be positive")
    if args.episodes != args.seed_end - args.seed_start + 1:
        raise ValueError("episodes must equal seed_end - seed_start + 1")
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    campaign = getattr(args, "campaign", None)
    if campaign is not None:
        if args.checkpoint is not None or args.tail_checkpoints is not None:
            raise ValueError("campaign cannot be combined with checkpoint/tail-checkpoints")
        checkpoints = campaign_checkpoint_paths(campaign)
    else:
        tail_checkpoints = args.tail_checkpoints
        if tail_checkpoints is None and args.checkpoint is None:
            configured = workflow.get("checkpoints", "latest")
            if str(configured).lower() == "latest":
                tail_checkpoints = workflow.get("tail_checkpoints")
        checkpoints = checkpoint_paths(
            args.checkpoint,
            workflow.get("checkpoints", "latest"),
            tail_checkpoints,
        )
    results: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        destination = output_root / candidate_label(checkpoint, args.suite)
        print(
            f"[scheduler] 候选进度={index}/{len(checkpoints)} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        if destination.exists():
            try:
                result = validate_candidate_output(
                    checkpoint,
                    destination,
                    args.episodes,
                    args.seed_start,
                    args.seed_end,
                    args.suite,
                )
                results.append(result)
                print("[scheduler] 已有完整结果，跳过", flush=True)
                continue
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                quarantine = quarantine_incomplete(destination)
                print(
                    f"[scheduler] 隔离不完整结果={quarantine} 原因={error}",
                    flush=True,
                )
        stale_pending = sorted(
            path
            for path in output_root.glob(f".{destination.name}.pending.*")
            if ".incomplete." not in path.name
        )
        for pending in stale_pending:
            quarantine = quarantine_incomplete(pending)
            print(f"[scheduler] 隔离上次中断的pending={quarantine}", flush=True)
        run_worker(
            worker_command(args, checkpoint, output_root),
            args.checkpoint_timeout_sec,
            checkpoint,
        )
        results.append(
            validate_candidate_output(
                checkpoint,
                destination,
                args.episodes,
                args.seed_start,
                args.seed_end,
                args.suite,
            )
        )

    ranking = sorted(
        results,
        key=lambda item: (
            bool(navigation_gate_failures(item["metrics"], gates)),
            *candidate_sort_key(item["metrics"]),
        ),
    )
    atomic_write_json(output_root / "candidate_ranking.json", ranking)
    best = {
        "schema_version": 1,
        "checkpoint": ranking[0]["checkpoint"],
        "evaluation": str(Path(ranking[0]["output"]) / "evaluation.json"),
        "selection_rule": [
            "navigation_gates_passed",
            "collision_rate",
            "-success_rate",
            "p95_cross_track_error_m",
            "mean_action_delta",
        ],
        "evaluation_suite": args.suite,
        "navigation_gates_passed": not navigation_gate_failures(
            ranking[0]["metrics"], gates
        ),
        "navigation_gate_failures": navigation_gate_failures(
            ranking[0]["metrics"], gates
        ),
        "production_eligible": False,
        "production_note": "仍需ONNX数值/时延和实车观测门禁",
    }
    atomic_write_json(output_root / "best_checkpoint.json", best)
    print(json.dumps(best, indent=2), flush=True)


def configure_policy(agent_cfg: Any, metadata: dict[str, Any], policy_name: str) -> None:
    """把 checkpoint 中的当前策略与 PPO 参数还原到 runner 配置。"""

    from car_rl_train.workflow import apply_object_config

    if policy_name != "split_fusion":
        raise ValueError(f"unsupported policy: {policy_name}")
    apply_object_config(agent_cfg.policy, metadata["policy"]["kwargs"], "policy")
    apply_object_config(
        agent_cfg.algorithm, metadata["algorithm"]["kwargs"], "algorithm"
    )


def worker_main(
    args: argparse.Namespace,
    workflow: dict[str, Any],
    simulation_app: Any,
) -> None:
    """在当前独立Isaac Sim进程内只评估一个checkpoint。"""

    import gymnasium as gym
    import torch

    import car_rl_train  # noqa: F401
    from car_rl_train.export.evaluate import EpisodeResult, summarize_episodes
    from car_rl_train.models import CarOnPolicyRunner
    from car_rl_train.runtime import configure_environment
    from car_rl_train.workflow import checkpoint_metadata
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    del simulation_app
    if args.worker_checkpoint is None:
        raise ValueError("worker缺少--worker-checkpoint")
    checkpoint = args.worker_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root = resolve_path(args.output_root)
    destination = output_root / candidate_label(checkpoint, args.suite)
    if destination.exists():
        raise FileExistsError(f"worker拒绝覆盖候选结果: {destination}")
    pending = output_root / f".{destination.name}.pending.{os.getpid()}"
    if pending.exists():
        raise FileExistsError(f"worker pending目录已存在: {pending}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint_metadata(payload, checkpoint)
    if metadata is None:
        raise ValueError(f"checkpoint不是当前split_fusion生产格式: {checkpoint}")
    policy_name = metadata["policy"]["name"]
    task = workflow.get("task", "Car-LocalNavigation-Eval-v0")
    device = args.device or workflow.get("device", "cuda:0")
    env_cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed_start
    configure_environment(
        env_cfg,
        workflow,
        training=False,
        calibration=args.calibration,
        checkpoint_metadata=metadata,
    )
    suite = suite_definition(workflow, args.suite)
    apply_suite_profile(env_cfg, suite)
    env_cfg.commands.navigation.seed_offset = args.seed_start
    env_cfg.commands.navigation.seed_stride = args.num_envs
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    configure_policy(agent_cfg, metadata, policy_name)

    env = None
    started = time.monotonic()
    try:
        env = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = CarOnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=None,
            device=device,
            checkpoint_metadata=metadata,
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
        observation = wrapped.get_observations()
        target_seeds = set(range(args.seed_start, args.seed_end + 1))
        if args.episodes != len(target_seeds):
            raise ValueError("episodes must equal seed_end - seed_start + 1")
        records: dict[int, dict[str, Any]] = {}
        simulation_observations: list[np.ndarray] = []
        deterministic_actions: list[np.ndarray] = []
        completed_cursor = 0
        last_report = 0
        max_steps = int(math.ceil(args.episodes / args.num_envs) + 4) * 450 * 2
        raw_env = env.unwrapped
        for _ in range(max_steps):
            with torch.inference_mode():
                action = policy(observation)
                observation, _, _, _ = wrapped.step(action)
            if len(deterministic_actions) < 10000:
                available = 10000 - len(deterministic_actions)
                deterministic_actions.extend(action.detach().cpu().numpy()[:available])
            if len(simulation_observations) < 10000:
                available = 10000 - len(simulation_observations)
                batch = raw_env._car_last_observation.detach().cpu().numpy()[:available]
                simulation_observations.extend(batch)
            completed = raw_env._car_completed_episodes
            for record in completed[completed_cursor:]:
                seed = int(record["seed"])
                if seed in target_seeds:
                    records.setdefault(seed, record)
            completed_cursor = len(completed)
            completed_count = len(records)
            if completed_count == args.episodes or completed_count >= last_report + 50:
                last_report = completed_count
                print(
                    f"[worker] checkpoint={checkpoint.name} "
                    f"episodes={completed_count}/{args.episodes} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            if target_seeds.issubset(records):
                break
        missing = sorted(target_seeds - set(records))
        if missing:
            raise RuntimeError(f"evaluation did not finish target seeds: {missing[:20]}")
        ordered = [records[seed] for seed in sorted(target_seeds)]
        metrics = summarize_episodes([EpisodeResult(**record) for record in ordered])
        observations = np.asarray(simulation_observations, dtype=np.float32)
        actions = np.asarray(deterministic_actions, dtype=np.float32)
        digest = hashlib.sha256()
        actor_state = {
            name: tensor
            for name, tensor in payload["model_state_dict"].items()
            if name.startswith("actor.")
        }
        for name, tensor in sorted(actor_state.items()):
            digest.update(name.removeprefix("actor.").encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        stop_spin = (actions[:, 0] <= -0.98) & (np.abs(actions[:, 1]) >= 0.98)
        metrics.update(
            {
                "actor_sha256": digest.hexdigest(),
                "deterministic_stop_spin_rate": float(np.mean(stop_spin)),
                "deterministic_action_component_std": [
                    float(value) for value in np.std(actions, axis=0)
                ],
                "checkpoint_training_diagnostics": metadata.get(
                    "training_diagnostics"
                ),
            }
        )
        metrics["evaluation_suite"] = args.suite
        failures = navigation_gate_failures(metrics, suite.get("gates", {}))
        metrics["navigation_gates_passed"] = not failures
        metrics["navigation_gate_failures"] = failures
    finally:
        if env is not None:
            env.close()

    pending.mkdir(parents=True, exist_ok=False)
    (pending / "episodes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    (pending / "evaluation.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(pending / "simulation_observations.npz", observation=observations)
    (pending / "simulation_observations.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "simulation",
                "production_eligible": False,
                "count": int(observations.shape[0]),
                "shape": [int(observations.shape[0]), 86],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    validate_candidate_output(
        checkpoint,
        pending,
        args.episodes,
        args.seed_start,
        args.seed_end,
        args.suite,
    )
    os.replace(pending, destination)
    print(f"[worker] 结果已发布={destination}", flush=True)


def main() -> None:
    """根据模式启动轻量调度器，或启动单checkpoint Isaac Sim worker。"""

    parser, workflow = build_common_parser()
    if "--worker" not in sys.argv[1:]:
        parser.add_argument("--device", default=workflow.get("device", "cuda:0"))
        parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
        parser.set_defaults(headless=bool(workflow.get("headless", True)))
        args = parser.parse_args()
        if args.campaign is None and args.checkpoint is None and args.suite == "all":
            pointer = ROOT / "logs" / "rsl_rl" / "car_local_navigation" / "campaigns" / "latest_campaign.json"
            if not pointer.is_file():
                raise FileNotFoundError("找不到最新训练 campaign，请先运行 06_train.py")
            args.campaign = str(pointer)
            if str(args.output_root) == "evaluations":
                selected = json.loads(pointer.read_text(encoding="utf-8"))["campaign"]
                args.output_root = str(ROOT / "evaluations" / Path(selected).stem)
        scheduler_main(args, workflow)
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=bool(workflow.get("headless", True)))
    args = parser.parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        worker_main(args, workflow, simulation_app)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise SystemExit(1)
    else:
        simulation_app.close()


if __name__ == "__main__":
    main()
