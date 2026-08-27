"""[必选] 在 Isaac Lab 中验证车辆驱动方向、动力学和目标到达行为。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--expect-goal", action="store_true")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--curriculum", default="configs/curriculum.yaml")
parser.add_argument("--calibration", default="calibration")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import car_rl_train  # noqa: E402,F401
from car_rl_train.runtime import configure_environment  # noqa: E402
from car_rl_train.tasks.local_navigation.observations import (  # noqa: E402
    _yaw_from_quaternion,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from car_rl_train.workflow import atomic_write_json  # noqa: E402


def publish(result: dict) -> None:
    """打印诊断结果，并在指定时原子写入 JSON。"""

    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    """施加两秒中性策略动作并校验真实位姿、轮速和车体速度。"""

    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    device = args.device or "cuda:0"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    # 基线动作的角速度严格为零；expect-goal 场景必须同样固定为直线，
    # 不能用课程中带曲率的路径去判定差速驱动物理是否正确。
    cfg = parse_env_cfg(
        "Car-LocalNavigation-v0", device=device, num_envs=args.num_envs
    )
    cfg.seed = 42
    cfg.commands.navigation.randomize_observations = False
    configure_environment(
        cfg,
        {"curriculum": args.curriculum, "calibration": args.calibration},
        training=True,
    )
    if args.expect_goal:
        straight = cfg.commands.navigation.stage_profiles[0]
        straight["max_curve_offset_m"] = 0.0
        straight["goal_yaw_offset_rad"] = 0.0
        cfg.commands.navigation.replay_previous_probability = 0.0
    env = gym.make("Car-LocalNavigation-v0", cfg=cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        start_xy = robot.data.root_pos_w[0, :2].clone()
        start_z = float(robot.data.root_pos_w[0, 2])
        start_yaw = _yaw_from_quaternion(robot.data.root_quat_w[:1])[0].clone()
        action = torch.zeros((args.num_envs, 2), device=device)
        terminations: dict[str, int] = {}
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            if bool(done.any()):
                manager = env.unwrapped.termination_manager
                for name in manager.active_terms:
                    count = int((manager.get_term(name) & done).sum().item())
                    if count:
                        terminations[name] = terminations.get(name, 0) + count
                if args.expect_goal and args.num_envs == 1:
                    break
        delta = robot.data.root_pos_w[0, :2] - start_xy
        forward = float(
            torch.cos(start_yaw) * delta[0] + torch.sin(start_yaw) * delta[1]
        )
        lateral = float(
            -torch.sin(start_yaw) * delta[0] + torch.cos(start_yaw) * delta[1]
        )
        elapsed = float(args.steps * env.unwrapped.step_dt)
        action_term = env.unwrapped.action_manager.get_term("drive")
        joint_velocity = robot.data.joint_vel[0, action_term._joint_ids]
        result = {
            "status": "ok",
            "device": str(device),
            "steps": int(args.steps),
            "num_envs": int(args.num_envs),
            "elapsed_sec": elapsed,
            "command_linear_mps": float(action_term.twist_command[0, 0]),
            "forward_displacement_m": forward,
            "lateral_displacement_m": lateral,
            "root_height_start_m": start_z,
            "root_height_end_m": float(robot.data.root_pos_w[0, 2]),
            "measured_forward_mps": float(robot.data.root_lin_vel_b[0, 0]),
            "wheel_joint_velocity_radps": [float(value) for value in joint_velocity],
            "wheel_internal_target_radps": [
                float(value) for value in action_term.wheel_feedback_target[0]
            ],
            "joint_ids": list(action_term._joint_ids),
            "joint_names": [robot.joint_names[index] for index in action_term._joint_ids],
            "applied_joint_torque_nm": [
                float(value) for value in robot.data.applied_torque[0, action_term._joint_ids]
            ],
            "body_names": list(robot.body_names),
            "body_heights_m": [
                float(value) for value in robot.data.body_pos_w[0, :, 2]
            ],
            "termination_counts": terminations,
        }
        # 前两秒包含落下约1 mm、轮端一阶响应与静摩擦起步；要求至少达到
        # 理想位移的20%，足以发现原先完全侧翻/卡轮的回归。
        if args.expect_goal:
            goals = terminations.get("goal_reached", 0)
            failures = sum(
                count
                for name, count in terminations.items()
                if name != "goal_reached"
            )
            if goals <= failures:
                raise RuntimeError(
                    "neutral straight policy did not reach the stage-0 goal: "
                    + json.dumps(result, ensure_ascii=False)
                )
            publish(result)
            return
        minimum = max(0.02, 0.20 * result["command_linear_mps"] * elapsed)
        if not math.isfinite(forward) or forward < minimum:
            raise RuntimeError(
                "positive wheel command did not move the car forward: "
                + json.dumps(result, ensure_ascii=False)
            )
        if not math.isfinite(lateral) or abs(lateral) > max(0.03, 0.5 * forward):
            raise RuntimeError(
                "straight wheel command produced excessive lateral motion: "
                + json.dumps(result, ensure_ascii=False)
            )
        publish(result)
    finally:
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
