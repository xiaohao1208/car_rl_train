"""有界且确定性的地图与路径指令生成器。"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Sequence

import numpy as np
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ...maps import DEFAULT_STAGE_CONFIGS, MapConfig, StageMapConfig, generate_navigation_map
from .geometry import MAP_RESOLUTION_M, PLANNING_CLEARANCE_M


def _default_stage_profiles() -> list[dict]:
    """把纯 Python 地图阶段配置转换为 Isaac configclass 可序列化结构。"""

    return [
        {
            "min_path_length_m": item.min_path_length_m,
            "max_path_length_m": item.max_path_length_m,
            "min_obstacles": item.min_obstacles,
            "max_obstacles": item.max_obstacles,
            "min_dynamic_obstacles": item.min_dynamic_obstacles,
            "max_dynamic_obstacles": item.max_dynamic_obstacles,
            "max_curve_offset_m": item.max_curve_offset_m,
            "goal_yaw_offset_rad": item.goal_yaw_offset_rad,
            "randomization_scale": item.randomization_scale,
        }
        for item in DEFAULT_STAGE_CONFIGS
    ]


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """将平面 yaw 转换为 Isaac 使用的 wxyz 四元数。"""

    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def _obstacle_component_centers(occupied: np.ndarray, resolution: float) -> list[tuple[float, float]]:
    """Approximate non-boundary connected components for the pooled colliders."""

    work = occupied.copy()
    work[[0, -1], :] = False
    work[:, [0, -1]] = False
    visited = np.zeros_like(work)
    centers: list[tuple[float, float]] = []
    for row, col in np.argwhere(work):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        component: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = current[0] + dr, current[1] + dc
                if (
                    0 <= nxt[0] < work.shape[0]
                    and 0 <= nxt[1] < work.shape[1]
                    and work[nxt]
                    and not visited[nxt]
                ):
                    visited[nxt] = True
                    stack.append(nxt)
        rows, cols = zip(*component)
        centers.append(
            (
                (0.5 * (min(cols) + max(cols)) + 0.5) * resolution - 3.0,
                (0.5 * (min(rows) + max(rows)) + 0.5) * resolution - 3.0,
            )
        )
    return centers


class NavigationCommand(CommandTerm):
    """为每个并行环境生成地图、路径、目标和动态障碍状态。"""

    cfg: "NavigationCommandCfg"

    def __init__(self, cfg: "NavigationCommandCfg", env):
        """预分配路径、占据栅格、进度和动态障碍张量。"""

        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.path_points_w = torch.zeros(
            (self.num_envs, cfg.max_path_points, 2), device=self.device
        )
        self.path_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.path_arc_lengths = torch.zeros_like(self.path_points_w[..., 0])
        self.occupancy_grid = torch.zeros(
            (self.num_envs, 120, 120), dtype=torch.bool, device=self.device
        )
        self.previous_progress = torch.zeros(self.num_envs, device=self.device)
        self.current_progress = torch.zeros(self.num_envs, device=self.device)
        self.last_progress_position = torch.zeros((self.num_envs, 2), device=self.device)
        self.stall_time = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_centers_w = torch.full(
            (self.num_envs, cfg.max_dynamic_obstacles, 2), 1.0e6, device=self.device
        )
        self.dynamic_velocities_w = torch.zeros_like(self.dynamic_centers_w)
        self.dynamic_active = torch.zeros(
            (self.num_envs, cfg.max_dynamic_obstacles), dtype=torch.bool, device=self.device
        )
        self.metrics["path_length_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["map_generation_attempts"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["map_fallback"] = torch.zeros(
            self.num_envs, device=self.device
        )
        # 地图与种子在 CPU 上生成。计数器也保留在 CPU，避免每个环境通过
        # Tensor.item() 强制同步整条 CUDA 流。
        self.episode_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device="cpu"
        )
        self.current_seed = torch.zeros(
            self.num_envs, dtype=torch.long, device="cpu"
        )
        self.episode_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_randomization_scale = torch.zeros(
            self.num_envs, device=self.device
        )
        self.lidar_force_refresh = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.localization_position_bias = torch.zeros((self.num_envs, 2), device=self.device)
        self.localization_yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self.localization_position_drift = torch.zeros_like(self.localization_position_bias)
        self.localization_yaw_drift = torch.zeros_like(self.localization_yaw_bias)
        self.lidar_bias = torch.zeros(self.num_envs, device=self.device)
        self.lidar_range_scale = torch.ones(self.num_envs, device=self.device)
        # 高频观测、碰撞和动态障碍计算复用这些不变量，避免逐步分配 CUDA 张量。
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.path_segment_indices = torch.arange(
            cfg.max_path_points - 1, device=self.device
        )[None, :]
        self.lidar_angles = (
            -math.pi
            + 2.0
            * math.pi
            * torch.arange(72, device=self.device, dtype=torch.float32)
            / 72.0
        )
        self.lidar_samples = torch.arange(
            0.05, 6.0001, 0.05, device=self.device
        )
        radius = float(cfg.dynamic_obstacle_radius_m + 0.02)
        self.dynamic_collision_offsets = torch.tensor(
            [
                [0.0, 0.0],
                [radius, 0.0],
                [-radius, 0.0],
                [0.0, radius],
                [0.0, -radius],
                [0.707 * radius, 0.707 * radius],
                [0.707 * radius, -0.707 * radius],
                [-0.707 * radius, 0.707 * radius],
                [-0.707 * radius, -0.707 * radius],
            ],
            device=self.device,
        )

    @property
    def stage_profile(self) -> StageMapConfig:
        """返回当前全局课程阶段的严格地图配置。"""

        profile = self.cfg.stage_profiles[int(self.cfg.curriculum_stage)]
        return StageMapConfig(**profile)

    @property
    def randomization_scale(self) -> float:
        """返回当前阶段实际启用的观测/动力学扰动比例。"""

        if not self.cfg.randomize_observations:
            return 0.0
        return float(self.stage_profile.randomization_scale)

    def randomization_scales(self, env_ids) -> torch.Tensor:
        """返回当前各回合的渐进随机化强度。"""

        if not self.cfg.randomize_observations:
            count = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
            return torch.zeros(count, device=self.device)
        return self.episode_randomization_scale[env_ids]

    @property
    def command(self) -> torch.Tensor:
        """返回各环境的目标 x、y 和 yaw。"""

        return self._command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """按独立种子重建指定环境任务，并重置机器人起始位姿。"""

        if isinstance(env_ids, slice):
            ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = list(env_ids)
        if not ids:
            return
        env_ids_tensor = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        robot = self._env.scene[self.cfg.asset_name]
        root_poses = robot.data.root_pose_w[env_ids_tensor].clone()
        root_velocities = torch.zeros((len(ids), 6), device=self.device)
        episode_indices = self.episode_counter[ids].tolist()
        seeds = [
            self.cfg.seed_offset
            + env_id
            + int(episode_index) * self.cfg.seed_stride
            for env_id, episode_index in zip(ids, episode_indices)
        ]
        current_stage = int(self.cfg.curriculum_stage)
        episode_stages = []
        for seed in seeds:
            rng = np.random.default_rng(seed ^ 0xC2B2AE35)
            replay = (
                current_stage > 0
                and self.cfg.replay_previous_probability > 0.0
                and float(rng.random()) < self.cfg.replay_previous_probability
            )
            episode_stages.append(current_stage - 1 if replay else current_stage)
        profiles = [StageMapConfig(**self.cfg.stage_profiles[stage]) for stage in episode_stages]
        navigation_maps = [
            generate_navigation_map(
                seed,
                stage=stage,
                config=MapConfig(
                    resolution_m=self.cfg.map_resolution_m,
                    inflation_radius_m=self.cfg.planning_clearance_m,
                    min_clearance_m=self.cfg.planning_clearance_m,
                ),
                stage_config=profile,
            )
            for seed, stage, profile in zip(seeds, episode_stages, profiles)
        ]
        for row, (env_id, seed, stage, profile, navigation_map) in enumerate(
            zip(ids, seeds, episode_stages, profiles, navigation_maps)
        ):
            self.current_seed[env_id] = seed
            self.episode_stage[env_id] = stage
            self.episode_randomization_scale[env_id] = float(profile.randomization_scale)
            self.occupancy_grid[env_id] = torch.as_tensor(
                navigation_map.occupied, dtype=torch.bool, device=self.device
            )
            origin = self._env.scene.env_origins[env_id, :2]
            local = np.asarray(navigation_map.path_xy, dtype=np.float32) - 3.0
            count = min(len(local), self.cfg.max_path_points)
            points = torch.as_tensor(local[:count], device=self.device) + origin
            self.path_points_w[env_id] = 0.0
            self.path_points_w[env_id, :count] = points
            if count < self.cfg.max_path_points:
                self.path_points_w[env_id, count:] = points[-1]
            self.path_lengths[env_id] = count
            segments = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
            self.path_arc_lengths[env_id] = 0.0
            self.path_arc_lengths[env_id, 1:count] = torch.cumsum(segments, dim=0)
            goal = points[-1]
            self._command[env_id] = torch.tensor(
                [goal[0], goal[1], navigation_map.goal_yaw],
                device=self.device,
            )
            start = points[0]
            heading = (
                math.atan2(float(points[1, 1] - points[0, 1]), float(points[1, 0] - points[0, 0]))
                if count > 1
                else 0.0
            )
            quaternion = _yaw_quaternion(heading)
            root_poses[row, :2] = start
            root_poses[row, 2] = self.cfg.spawn_height_m
            root_poses[row, 3:7] = torch.tensor(quaternion, device=self.device)
            self.metrics["path_length_m"][env_id] = self.path_arc_lengths[env_id, count - 1]
            self.metrics["map_generation_attempts"][env_id] = float(
                navigation_map.generation_attempts
            )
            self.metrics["map_fallback"][env_id] = float(
                navigation_map.fallback_used
            )
            self.dynamic_centers_w[env_id] = 1.0e6
            self.dynamic_velocities_w[env_id] = 0.0
            self.dynamic_active[env_id] = False
            rng = np.random.default_rng(seed ^ 0x5A17)
            dynamic_count = int(
                rng.integers(
                    profile.min_dynamic_obstacles,
                    profile.max_dynamic_obstacles + 1,
                )
            )
            dynamic_count = min(self.cfg.max_dynamic_obstacles, dynamic_count)
            if dynamic_count:
                free_rows_cols = np.argwhere(~navigation_map.inflated)
                rng.shuffle(free_rows_cols)
                chosen: list[np.ndarray] = []
                for cell in free_rows_cols:
                    candidate = np.array(
                        [(cell[1] + 0.5) * 0.05 - 3.0, (cell[0] + 0.5) * 0.05 - 3.0],
                        dtype=np.float32,
                    )
                    path_clear = np.min(np.linalg.norm(local - candidate, axis=1)) > 0.60
                    separated = all(
                        np.linalg.norm(existing - candidate) > 0.50
                        for existing in chosen
                    )
                    if path_clear and separated:
                        chosen.append(candidate)
                    if len(chosen) == dynamic_count:
                        break
                for obstacle_index, center in enumerate(chosen):
                    angle = rng.uniform(-math.pi, math.pi)
                    speed = rng.uniform(0.08, 0.18) * float(
                        profile.randomization_scale
                    )
                    self.dynamic_centers_w[env_id, obstacle_index] = (
                        torch.as_tensor(center, device=self.device) + origin
                    )
                    self.dynamic_velocities_w[env_id, obstacle_index] = torch.tensor(
                        [math.cos(angle) * speed, math.sin(angle) * speed],
                        device=self.device,
                    )
                    self.dynamic_active[env_id, obstacle_index] = True
        self.episode_counter[ids] += 1
        scale = self.randomization_scales(env_ids_tensor)
        count = len(ids)
        if torch.any(scale > 0.0):
            self.localization_position_bias[env_ids_tensor] = (
                torch.randn((count, 2), device=self.device)
                * self.cfg.localization_position_std_m
                * scale[:, None]
            )
            self.localization_yaw_bias[env_ids_tensor] = (
                torch.randn(count, device=self.device)
                * self.cfg.localization_yaw_std_rad
                * scale
            )
            self.localization_position_drift[env_ids_tensor] = (
                torch.empty((count, 2), device=self.device).uniform_(-1.0, 1.0)
                * self.cfg.localization_position_drift_mps
                * scale[:, None]
            )
            self.localization_yaw_drift[env_ids_tensor] = (
                torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                * self.cfg.localization_yaw_drift_radps
                * scale
            )
            self.lidar_bias[env_ids_tensor] = (
                torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                * self.cfg.lidar_bias_max_m
                * scale
            )
            self.lidar_range_scale[env_ids_tensor] = 1.0 + (
                torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                * self.cfg.lidar_range_scale_max_error
                * scale
            )
        else:
            self.localization_position_bias[env_ids_tensor] = 0.0
            self.localization_yaw_bias[env_ids_tensor] = 0.0
            self.localization_position_drift[env_ids_tensor] = 0.0
            self.localization_yaw_drift[env_ids_tensor] = 0.0
            self.lidar_bias[env_ids_tensor] = 0.0
            self.lidar_range_scale[env_ids_tensor] = 1.0
        robot.write_root_pose_to_sim(root_poses, env_ids=env_ids_tensor)
        robot.write_root_velocity_to_sim(root_velocities, env_ids=env_ids_tensor)
        self.previous_progress[env_ids_tensor] = 0.0
        self.current_progress[env_ids_tensor] = 0.0
        self.last_progress_position[env_ids_tensor] = root_poses[:, :2]
        self.stall_time[env_ids_tensor] = 0.0
        self.lidar_force_refresh[env_ids_tensor] = True
        for name, fill in (
            ("_car_lidar_age", 0.0),
            ("_car_lidar_stale", False),
            ("_car_lidar_timeout", False),
            ("_car_hard_lidar_fault", False),
            ("_car_goal_counter", 0),
            ("_car_near_goal", False),
            ("_car_previous_goal_yaw_error", 0.0),
        ):
            value = getattr(self._env, name, None)
            if value is not None:
                value[env_ids_tensor] = fill

    def _update_command(self) -> None:
        """推进互惠动态障碍，并在墙体、边界、终点和车辆前反弹。"""

        self.localization_position_bias.add_(
            self.localization_position_drift * self._env.step_dt
        )
        self.localization_yaw_bias.add_(
            self.localization_yaw_drift * self._env.step_dt
        )
        if self.stage_profile.max_dynamic_obstacles <= 0:
            return
        proposed = self.dynamic_centers_w + self.dynamic_velocities_w * self._env.step_dt
        origins = self._env.scene.env_origins[:, None, :2]
        local = proposed - origins
        hit_boundary = (torch.abs(local) > 2.65).any(dim=-1)
        samples = local[:, :, None, :] + self.dynamic_collision_offsets[None, None, :, :]
        cols = torch.clamp(((samples[..., 0] + 3.0) / 0.05).long(), 0, 119)
        rows = torch.clamp(((samples[..., 1] + 3.0) / 0.05).long(), 0, 119)
        hit_static = self.occupancy_grid[
            self.env_ids[:, None, None], rows, cols
        ].any(dim=-1)
        hit_goal = torch.linalg.vector_norm(
            proposed - self._command[:, None, :2], dim=-1
        ) < 0.45
        robot_xy = self._env.scene[self.cfg.asset_name].data.root_pos_w[:, None, :2]
        robot_delta = proposed - robot_xy
        approaching_robot = torch.sum(
            self.dynamic_velocities_w * robot_delta, dim=-1
        ) < 0.0
        hit_robot_yield_zone = (
            torch.linalg.vector_norm(robot_delta, dim=-1)
            < self.cfg.dynamic_robot_yield_radius_m
        ) & approaching_robot
        blocked = (
            hit_boundary | hit_static | hit_goal | hit_robot_yield_zone
        ) & self.dynamic_active
        self.dynamic_velocities_w.copy_(
            torch.where(blocked[..., None], -self.dynamic_velocities_w, self.dynamic_velocities_w)
        )
        proposed = self.dynamic_centers_w + self.dynamic_velocities_w * self._env.step_dt
        self.dynamic_centers_w.copy_(
            torch.where(self.dynamic_active[..., None], proposed, self.dynamic_centers_w)
        )

    def _update_metrics(self) -> None:
        """更新机器人到目标的真值距离指标。"""

        robot = self._env.scene[self.cfg.asset_name]
        distance = torch.linalg.vector_norm(
            self._command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
        )
        if "goal_distance_m" not in self.metrics:
            self.metrics["goal_distance_m"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["goal_distance_m"].copy_(distance)


@configclass
class NavigationCommandCfg(CommandTermCfg):
    """地图种子、路径容量、动态障碍和观测扰动配置。"""

    class_type: type[CommandTerm] = NavigationCommand
    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    seed_offset: int = 0
    seed_stride: int = 100000
    curriculum_stage: int = 0
    stage_profiles: list[dict] = _default_stage_profiles()
    replay_previous_probability: float = 0.20
    map_resolution_m: float = MAP_RESOLUTION_M
    planning_clearance_m: float = PLANNING_CLEARANCE_M
    resume_stage_episode_total: int = 0
    max_path_points: int = 128
    # URDF 的 articulation 根是 base_footprint；轮子/脚轮底面约位于根坐标
    # z=-1 mm。只留 2 mm 数值净空，避免每次 reset 从 8 cm 高处跌落并侧翻。
    spawn_height_m: float = 0.002
    max_dynamic_obstacles: int = 4
    dynamic_obstacle_radius_m: float = 0.16
    # 模拟会主动避让已停车车辆的行人/移动机器人；非合作高速目标由
    # 实车急停安全链处理，不纳入局部导航策略的能力声明。
    dynamic_robot_yield_radius_m: float = 0.45
    randomize_observations: bool = True
    lidar_noise_std_m: float = 0.015
    lidar_bias_max_m: float = 0.01
    lidar_range_scale_max_error: float = 0.01
    lidar_invalid_ray_probability: float = 0.002
    lidar_frame_drop_probability: float = 0.01
    localization_position_std_m: float = 0.03
    localization_yaw_std_rad: float = 0.035
    # 最大漂移约为 3 cm、0.03 rad/分钟；更大的无界漂移无法从 86 维
    # 观测中辨识，也不代表部署端经过定位健康检查后的输入。
    localization_position_drift_mps: float = 0.0005
    localization_yaw_drift_radps: float = 0.0005
