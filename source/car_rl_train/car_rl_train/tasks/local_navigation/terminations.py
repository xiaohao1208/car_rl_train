"""目标、路径偏离、停滞、碰撞和输入安全终止条件。"""

from __future__ import annotations

import torch

from .observations import _yaw_from_quaternion
from .geometry import FOOTPRINT_POINTS


GOAL_POSITION_TOLERANCE_M = 0.08
GOAL_YAW_TOLERANCE_RAD = 0.174532925


def goal_reached(env, command_name: str = "navigation") -> torch.Tensor:
    """位置与偏航连续五步同时达标时判定成功。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    distance = torch.linalg.vector_norm(command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    yaw = _yaw_from_quaternion(robot.data.root_quat_w)
    yaw_error = torch.abs(
        torch.atan2(
            torch.sin(command.command[:, 2] - yaw),
            torch.cos(command.command[:, 2] - yaw),
        )
    )
    condition = (distance <= GOAL_POSITION_TOLERANCE_M) & (
        yaw_error <= GOAL_YAW_TOLERANCE_RAD
    )
    counter = getattr(env, "_car_goal_counter", None)
    if counter is None:
        counter = env._car_goal_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counter.copy_(torch.where(condition, counter + 1, torch.zeros_like(counter)))
    return counter >= 5


def path_deviation(env) -> torch.Tensor:
    """横向路径误差超过半米时终止回合。"""

    return env._car_cross_track_error > 0.5


def stalled(env, command_name: str = "navigation") -> torch.Tensor:
    """五秒内未产生足够位移且尚未到达目标时判定停滞。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    moved = torch.linalg.vector_norm(
        robot.data.root_pos_w[:, :2] - command.last_progress_position, dim=-1
    )
    progressed = moved >= 0.04
    command.last_progress_position.copy_(
        torch.where(progressed[:, None], robot.data.root_pos_w[:, :2], command.last_progress_position)
    )
    command.stall_time.copy_(
        torch.where(progressed, torch.zeros_like(command.stall_time), command.stall_time + env.step_dt)
    )
    distance = torch.linalg.vector_norm(command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return (command.stall_time >= 5.0) & (
        distance > GOAL_POSITION_TOLERANCE_M
    )


def invalid_input(env) -> torch.Tensor:
    """观测出现非有限值或雷达整帧超时超过 0.5 秒时终止。"""

    invalid = getattr(env, "_car_invalid_observation", None)
    if invalid is None:
        invalid = env._car_invalid_observation = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    timeout = getattr(env, "_car_lidar_timeout", None)
    if timeout is None:
        timeout = env._car_lidar_timeout = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    return invalid | timeout


def collision(env, command_name: str = "navigation") -> torch.Tensor:
    """用车辆足迹与静态栅格、动态圆障碍的重叠计算真值碰撞。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    yaw = _yaw_from_quaternion(robot.data.root_quat_w)
    footprint = getattr(env, "_car_footprint_points", None)
    if footprint is None:
        footprint = env._car_footprint_points = torch.tensor(
            FOOTPRINT_POINTS, device=env.device
        )
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    x = (
        robot.data.root_pos_w[:, None, 0]
        + cosine[:, None] * footprint[None, :, 0]
        - sine[:, None] * footprint[None, :, 1]
    )
    y = (
        robot.data.root_pos_w[:, None, 1]
        + sine[:, None] * footprint[None, :, 0]
        + cosine[:, None] * footprint[None, :, 1]
    )
    cols = torch.clamp(
        ((x - env.scene.env_origins[:, None, 0] + 3.0) / 0.05).long(), 0, 119
    )
    rows = torch.clamp(
        ((y - env.scene.env_origins[:, None, 1] + 3.0) / 0.05).long(), 0, 119
    )
    env_ids = command.env_ids[:, None]
    occupancy_hit = command.occupancy_grid[env_ids, rows, cols].any(dim=-1)
    dynamic_delta = (
        torch.stack((x, y), dim=-1)[:, :, None, :]
        - command.dynamic_centers_w[:, None, :, :]
    )
    dynamic_hit = (
        torch.linalg.vector_norm(dynamic_delta, dim=-1)
        <= command.cfg.dynamic_obstacle_radius_m
    ) & command.dynamic_active[:, None, :]
    dynamic_hit = dynamic_hit.any(dim=(1, 2))
    env._car_static_collision = occupancy_hit
    env._car_dynamic_collision = dynamic_hit
    return occupancy_hit | dynamic_hit
