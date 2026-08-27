"""固定实现计划中定义的各项奖励函数。"""

from __future__ import annotations

import torch

from .observations import _sample_lookahead, _yaw_from_quaternion
from .geometry import FOOTPRINT_RADIUS_M, PATH_SAFETY_MARGIN_M


def path_progress(env, command_name: str = "navigation") -> torch.Tensor:
    """返回贴近路径时的正向进度速率；倒退始终完整扣分。"""

    command = env.command_manager.get_term(command_name)
    delta = torch.clamp(
        command.current_progress - command.previous_progress, -0.05, 0.05
    )
    cross_track = torch.clamp(env._car_cross_track_error, min=0.0)
    on_path_scale = torch.exp(-8.0 * cross_track**2)
    shaped_delta = torch.where(delta > 0.0, delta * on_path_scale, delta)
    return shaped_delta / float(env.step_dt)


def terminal_event(env, term_name: str) -> torch.Tensor:
    """返回不被 RewardManager 的 step_dt 稀释的离散终止事件。"""

    value = env.termination_manager.get_term(term_name)
    return value.to(dtype=torch.float32) / float(env.step_dt)


def cross_track_error(env) -> torch.Tensor:
    """返回路径横向绝对误差，为小偏差也提供稳定纠偏梯度。"""

    return torch.clamp(env._car_cross_track_error, 0.0, 1.0)


def heading_error_cost(env, command_name: str = "navigation") -> torch.Tensor:
    """远端对准路径前视点，近端切换为最终目标朝向。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    yaw = _yaw_from_quaternion(robot.data.root_quat_w)
    lookahead = _sample_lookahead(command, command.current_progress)[:, 0]
    path_direction = torch.atan2(
        lookahead[:, 1] - robot.data.root_pos_w[:, 1],
        lookahead[:, 0] - robot.data.root_pos_w[:, 0],
    )
    distance = torch.linalg.vector_norm(
        command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    desired = torch.where(distance <= 0.35, command.command[:, 2], path_direction)
    error = torch.atan2(torch.sin(desired - yaw), torch.cos(desired - yaw))
    return 1.0 - torch.cos(error)


def obstacle_proximity_cost(env) -> torch.Tensor:
    """按车辆足迹外缘到障碍物的净空计算近障惩罚。"""

    footprint_clearance = env._car_min_clearance - FOOTPRINT_RADIUS_M
    return torch.clamp(
        (PATH_SAFETY_MARGIN_M - footprint_clearance) / PATH_SAFETY_MARGIN_M,
        min=0.0,
    ) ** 2


def near_goal_yaw_progress(env, command_name: str = "navigation") -> torch.Tensor:
    """只奖励近目标后朝向误差的实际减少，避免原地等待刷奖励。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    distance = torch.linalg.vector_norm(
        command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    yaw = _yaw_from_quaternion(robot.data.root_quat_w)
    error = torch.abs(
        torch.atan2(
            torch.sin(command.command[:, 2] - yaw),
            torch.cos(command.command[:, 2] - yaw),
        )
    )
    previous = getattr(env, "_car_previous_goal_yaw_error", None)
    if previous is None:
        previous = env._car_previous_goal_yaw_error = error.clone()
    was_near = getattr(env, "_car_near_goal", None)
    if was_near is None:
        was_near = env._car_near_goal = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    near = distance <= 0.35
    delta = torch.clamp(previous - error, -0.20, 0.20)
    result = torch.where(near & was_near, delta / float(env.step_dt), 0.0)
    previous.copy_(error)
    was_near.copy_(near)
    return result


def action_delta_squared(env) -> torch.Tensor:
    """惩罚相邻策略步之间的动作突变。"""

    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)


def angular_action_squared(env) -> torch.Tensor:
    """惩罚过大的归一化角速度动作。"""

    return env.action_manager.action[:, 1] ** 2


def stopped_away_from_goal(env, command_name: str = "navigation") -> torch.Tensor:
    """车辆远离目标且线速度过低时施加停车惩罚。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    distance = torch.linalg.vector_norm(command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return ((torch.abs(robot.data.root_lin_vel_b[:, 0]) < 0.01) & (distance > 0.25)).float()
