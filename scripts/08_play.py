"""[可选] 使用 checkpoint 元数据重建策略并执行确定性回放。"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

from car_rl_train.workflow import load_workflow_config, resolve_checkpoint_selection

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", type=Path, default=ROOT / "configs" / "play.yaml")
known, _ = pre_parser.parse_known_args()
workflow = load_workflow_config(known.config)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(parents=[pre_parser])
parser.add_argument("--checkpoint", default=workflow.get("checkpoint", "latest"))
parser.add_argument("--num_envs", type=int, default=workflow.get("num_envs", 16))
parser.add_argument("--steps", type=int, default=workflow.get("steps", 2000))
parser.add_argument("--calibration", default=workflow.get("calibration"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=bool(workflow.get("headless", False)))
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import car_rl_train  # noqa: E402,F401
from car_rl_train.models import CarOnPolicyRunner  # noqa: E402
from car_rl_train.runtime import configure_environment  # noqa: E402
from car_rl_train.workflow import apply_object_config, checkpoint_metadata  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def main() -> None:
    """加载当前生产 checkpoint，并在关闭训练扰动的环境执行确定性策略。"""

    checkpoint, _ = resolve_checkpoint_selection(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint_metadata(payload, checkpoint)
    if metadata is None:
        raise ValueError("checkpoint不是当前split_fusion生产格式")
    task = workflow.get("task", "Car-LocalNavigation-Play-v0")
    device = args.device or workflow.get("device", "cuda:0")
    env_cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    env_cfg.seed = int(metadata.get("seed") or 42)
    configure_environment(
        env_cfg,
        workflow,
        training=False,
        calibration=args.calibration,
        checkpoint_metadata=metadata,
    )
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    apply_object_config(agent_cfg.policy, metadata["policy"]["kwargs"], "policy")
    apply_object_config(
        agent_cfg.algorithm, metadata["algorithm"]["kwargs"], "algorithm"
    )
    env = None
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
        for _ in range(args.steps):
            with torch.inference_mode():
                action = policy(observation)
                observation, _, _, _ = wrapped.step(action)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise SystemExit(1)
    else:
        simulation_app.close()
