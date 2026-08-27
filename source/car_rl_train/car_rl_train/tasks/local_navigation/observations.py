"""在 Isaac Lab 内严格构造向量化 86 维控制器观测。"""

from __future__ import annotations

import torch


def _yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """批量从 wxyz 四元数提取平面 yaw。"""

    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _project_path(command, robot_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将每台车投影到完整路径折线，返回弧长进度和横向误差。"""

    points = command.path_points_w
    starts, ends = points[:, :-1], points[:, 1:]
    segments = ends - starts
    lengths_sq = torch.sum(segments * segments, dim=-1)
    relative = robot_xy[:, None, :] - starts
    ratios = torch.clamp(
        torch.sum(relative * segments, dim=-1) / torch.clamp(lengths_sq, min=1.0e-12),
        0.0,
        1.0,
    )
    projections = starts + ratios[..., None] * segments
    distance_sq = torch.sum((robot_xy[:, None, :] - projections) ** 2, dim=-1)
    indices = command.path_segment_indices
    valid = indices < (command.path_lengths - 1)[:, None]
    distance_sq = torch.where(valid, distance_sq, torch.full_like(distance_sq, torch.inf))
    nearest = torch.argmin(distance_sq, dim=1)
    env_ids = command.env_ids
    nearest_ratio = ratios[env_ids, nearest]
    segment_lengths = torch.sqrt(torch.clamp(lengths_sq[env_ids, nearest], min=0.0))
    progress = command.path_arc_lengths[env_ids, nearest] + nearest_ratio * segment_lengths
    return progress, torch.sqrt(distance_sq[env_ids, nearest]), nearest


def _sample_lookahead(command, progress: torch.Tensor) -> torch.Tensor:
    """沿路径弧长批量采样 0.3、0.6 和 1.0 米前视点。"""

    cumulative = command.path_arc_lengths
    env_ids = command.env_ids
    total = cumulative[env_ids, command.path_lengths - 1]
    results = []
    for distance in (0.3, 0.6, 1.0):
        target = torch.minimum(progress + distance, total)
        after = torch.argmax((cumulative >= target[:, None]).to(torch.int64), dim=1)
        after = torch.maximum(after, torch.ones_like(after))
        before = after - 1
        before_arc = cumulative[env_ids, before]
        after_arc = cumulative[env_ids, after]
        ratio = (target - before_arc) / torch.clamp(after_arc - before_arc, min=1.0e-9)
        point = command.path_points_w[env_ids, before] + ratio[:, None] * (
            command.path_points_w[env_ids, after] - command.path_points_w[env_ids, before]
        )
        results.append(point)
    return torch.stack(results, dim=1)


def controller_observation(env, command_name: str = "navigation") -> torch.Tensor:
    """构造与 ROS 2 C++ 控制器严格一致的 86 维策略观测。"""

    robot = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    root_xy = robot.data.root_pos_w[:, :2]
    yaw = _yaw_from_quaternion(robot.data.root_quat_w)
    progress, cross_track, _ = _project_path(command, root_xy)
    perceived_xy = root_xy + command.localization_position_bias
    perceived_yaw = yaw + command.localization_yaw_bias
    perceived_progress, _, _ = _project_path(command, perceived_xy)
    lookahead = _sample_lookahead(command, perceived_progress)
    cosine, sine = torch.cos(perceived_yaw), torch.sin(perceived_yaw)
    delta = lookahead - perceived_xy[:, None, :]
    path_x = cosine[:, None] * delta[..., 0] + sine[:, None] * delta[..., 1]
    path_y = -sine[:, None] * delta[..., 0] + cosine[:, None] * delta[..., 1]
    path_values = torch.stack((path_x, path_y), dim=-1).reshape(env.num_envs, 6)
    path_values = torch.clamp(path_values, -1.5, 1.5) / 1.5

    goal_delta = command.command[:, :2] - perceived_xy
    goal_x = cosine * goal_delta[:, 0] + sine * goal_delta[:, 1]
    goal_y = -sine * goal_delta[:, 0] + cosine * goal_delta[:, 1]
    yaw_error = torch.atan2(
        torch.sin(command.command[:, 2] - perceived_yaw),
        torch.cos(command.command[:, 2] - perceived_yaw),
    )

    # 每个环境的程序化几何由紧凑占据栅格表示；在 GPU 上进行射线步进，
    # 可在保持 72 束契约一致的同时扩展到大量相互独立的环境。
    angles = command.lidar_angles
    world_angles = yaw[:, None] + angles[None, :]
    samples = command.lidar_samples
    x = root_xy[:, None, None, 0] + torch.cos(world_angles)[:, :, None] * samples
    y = root_xy[:, None, None, 1] + torch.sin(world_angles)[:, :, None] * samples
    local_x = x - env.scene.env_origins[:, None, None, 0] + 3.0
    local_y = y - env.scene.env_origins[:, None, None, 1] + 3.0
    cols = torch.clamp((local_x / 0.05).long(), 0, 119)
    rows = torch.clamp((local_y / 0.05).long(), 0, 119)
    env_index = command.env_ids[:, None, None]
    occupied_samples = command.occupancy_grid[env_index, rows, cols]
    first_hit = torch.argmax(occupied_samples.to(torch.int64), dim=-1)
    has_hit = occupied_samples.any(dim=-1)
    grid_distance = torch.where(
        has_hit,
        samples[first_hit],
        torch.full_like(first_hit, 6.0, dtype=torch.float32),
    )
    # 动态圆形障碍采用批量解析式射线—圆求交，避免为每个障碍创建独立射线网格。
    relative_centers = command.dynamic_centers_w - root_xy[:, None, :]
    directions = torch.stack((torch.cos(world_angles), torch.sin(world_angles)), dim=-1)
    projected = torch.einsum("nkd,nrd->nrk", relative_centers, directions)
    center_sq = torch.sum(relative_centers * relative_centers, dim=-1)[:, None, :]
    perpendicular_sq = center_sq - projected * projected
    radius_sq = command.cfg.dynamic_obstacle_radius_m**2
    valid_circle = (
        command.dynamic_active[:, None, :]
        & (projected > 0.0)
        & (perpendicular_sq <= radius_sq)
    )
    circle_distance = projected - torch.sqrt(torch.clamp(radius_sq - perpendicular_sq, min=0.0))
    circle_distance = torch.where(
        valid_circle,
        circle_distance,
        torch.full_like(circle_distance, 6.0),
    )
    dynamic_distance = torch.clamp(circle_distance.min(dim=-1).values, 0.05, 6.0)
    distances = torch.minimum(grid_distance, dynamic_distance)
    cache = getattr(env, "_car_lidar_cache", None)
    if cache is None:
        cache = env._car_lidar_cache = distances.clone()
        env._car_lidar_age = torch.zeros(env.num_envs, device=env.device)
    scheduled_refresh = (env.episode_length_buf % 3) == 0
    forced_refresh = command.lidar_force_refresh
    refresh = scheduled_refresh | forced_refresh
    scale = (
        command.episode_randomization_scale
        if command.cfg.randomize_observations
        else torch.zeros(env.num_envs, device=env.device)
    )
    if command.randomization_scale > 0.0:
        distances = (
            distances * command.lidar_range_scale[:, None]
            + command.lidar_bias[:, None]
            + torch.randn_like(distances) * command.cfg.lidar_noise_std_m * scale[:, None]
        )
        frame_drop = (
            torch.rand(env.num_envs, device=env.device)
            < command.cfg.lidar_frame_drop_probability * scale
        )
        refresh &= ~(frame_drop & ~forced_refresh)
        invalid_ray = (
            torch.rand_like(distances)
            < command.cfg.lidar_invalid_ray_probability * scale[:, None]
        )
        distances = torch.where(invalid_ray, torch.full_like(distances, 6.0), distances)
    distances = torch.clamp(distances, 0.05, 6.0)
    cache.copy_(torch.where(refresh[:, None], distances, cache))
    command.lidar_force_refresh.copy_(command.lidar_force_refresh & ~refresh)
    env._car_lidar_age.copy_(
        torch.where(refresh, torch.zeros_like(env._car_lidar_age), env._car_lidar_age + env.step_dt)
    )
    # 模拟丢帧代表可恢复的短时传输抖动，只复用上一帧并记录陈旧时间。
    # 硬件超时由显式 _car_hard_lidar_fault 注入；不能让随机丢帧制造
    # 与策略无关、且策略无法从86维契约中恢复的终止回合。
    env._car_lidar_stale = env._car_lidar_age > 0.5
    hard_fault = getattr(env, "_car_hard_lidar_fault", None)
    if hard_fault is None:
        hard_fault = env._car_hard_lidar_fault = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._car_lidar_timeout = hard_fault.clone()
    distances = cache
    lidar = (distances - 0.05) / 5.95
    if lidar.shape[-1] != 72:
        raise RuntimeError(f"contract lidar must have 72 rays, got {lidar.shape}")

    action_term = env.action_manager.get_term("drive")
    previous = env.action_manager.prev_action
    observation = torch.cat(
        (
            lidar,
            path_values,
            torch.clamp(goal_x, -3.0, 3.0)[:, None] / 3.0,
            torch.clamp(goal_y, -3.0, 3.0)[:, None] / 3.0,
            torch.sin(yaw_error)[:, None],
            torch.cos(yaw_error)[:, None],
            torch.clamp(robot.data.root_lin_vel_b[:, 0], -0.25, 0.25)[:, None] / 0.25,
            torch.clamp(robot.data.root_ang_vel_b[:, 2], -1.0, 1.0)[:, None],
            torch.clamp(previous, -1.0, 1.0),
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    command.previous_progress.copy_(command.current_progress)
    command.current_progress.copy_(progress)
    env._car_cross_track_error = cross_track
    env._car_min_clearance = torch.min(distances, dim=-1).values
    env._car_invalid_observation = ~torch.isfinite(observation).all(dim=-1)
    if observation.shape != (env.num_envs, 86) or env._car_invalid_observation.any():
        raise RuntimeError("invalid controller observation")
    env._car_last_observation = observation
    return observation
