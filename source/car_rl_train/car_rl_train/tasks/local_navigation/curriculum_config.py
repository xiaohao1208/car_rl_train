"""课程 YAML 的严格校验以及训练/评估环境注入。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import yaml

from .geometry import (
    FOOTPRINT_HALF_LENGTH_M,
    FOOTPRINT_HALF_WIDTH_M,
    MAP_RESOLUTION_M,
    PATH_SAFETY_MARGIN_M,
    PLANNING_CLEARANCE_M,
)


def load_curriculum_config(path: str | Path) -> dict[str, Any]:
    """加载通用多阶段课程配置并拒绝缺项、乱序和无效数值。"""

    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 6:
        raise ValueError("curriculum config must use schema_version=6")
    geometry = value.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("curriculum geometry contract is missing")
    half_length = float(geometry.get("footprint_half_length_m", 0.0))
    half_width = float(geometry.get("footprint_half_width_m", 0.0))
    margin = float(geometry.get("path_safety_margin_m", 0.0))
    resolution = float(geometry.get("map_resolution_m", 0.0))
    clearance = float(geometry.get("planning_clearance_m", 0.0))
    expected_clearance = (
        math.ceil((math.hypot(half_length, half_width) + margin) / resolution)
        * resolution
        if resolution > 0.0
        else 0.0
    )
    if (
        half_length <= 0.0
        or half_width <= 0.0
        or margin <= 0.0
        or resolution <= 0.0
        or not math.isclose(clearance, expected_clearance, abs_tol=1.0e-9)
        or not math.isclose(half_length, FOOTPRINT_HALF_LENGTH_M, abs_tol=1.0e-9)
        or not math.isclose(half_width, FOOTPRINT_HALF_WIDTH_M, abs_tol=1.0e-9)
        or not math.isclose(margin, PATH_SAFETY_MARGIN_M, abs_tol=1.0e-9)
        or not math.isclose(resolution, MAP_RESOLUTION_M, abs_tol=1.0e-9)
        or not math.isclose(clearance, PLANNING_CLEARANCE_M, abs_tol=1.0e-9)
    ):
        raise ValueError("curriculum geometry contract is inconsistent")
    stages = value.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) < 2
        or [item.get("stage") for item in stages] != list(range(len(stages)))
    ):
        raise ValueError("curriculum config must contain consecutive ordered stages")
    for item in stages:
        path_range = item.get("path_length_m")
        obstacle_range = item.get("static_obstacles")
        dynamic_range = item.get("dynamic_obstacles")
        if (
            not isinstance(path_range, list)
            or len(path_range) != 2
            or float(path_range[0]) <= 0.0
            or float(path_range[1]) < float(path_range[0])
            or not isinstance(obstacle_range, list)
            or len(obstacle_range) != 2
            or int(obstacle_range[0]) < 0
            or int(obstacle_range[1]) < int(obstacle_range[0])
            or not isinstance(dynamic_range, list)
            or len(dynamic_range) != 2
            or int(dynamic_range[0]) < 0
            or int(dynamic_range[1]) < int(dynamic_range[0])
            or not 0.0 <= float(item.get("max_curve_offset_m", -1.0)) <= 1.0
            or not 0.0 <= float(item.get("goal_yaw_offset_deg", -1.0)) <= 180.0
            or not 0.0 <= float(item.get("randomization_scale", -1.0)) <= 1.0
            or not 0.0 <= float(item.get("success_rate", -1.0)) <= 1.0
            or not 0.0 <= float(item.get("max_collision_rate", -1.0)) <= 1.0
            or int(item.get("min_stage_episodes", 0)) < int(value.get("evaluation_episodes", 0))
        ):
            raise ValueError(f"invalid curriculum stage: {item.get('stage')}")
    yaw_offsets = [float(item["goal_yaw_offset_deg"]) for item in stages]
    scales = [float(item["randomization_scale"]) for item in stages]
    if yaw_offsets != sorted(yaw_offsets) or yaw_offsets[-1] != 180.0:
        raise ValueError("goal yaw offsets must increase to 180 degrees")
    if scales != sorted(scales) or scales[-1] != 1.0:
        raise ValueError("randomization scales must increase to 1.0")
    if int(value.get("evaluation_episodes", 0)) <= 0:
        raise ValueError("evaluation_episodes must be positive")
    if int(value.get("consecutive_passes", 0)) <= 0:
        raise ValueError("consecutive_passes must be positive")
    if int(value.get("initial_stage", -1)) not in range(len(stages)):
        raise ValueError("initial_stage is outside configured stages")
    replay = float(value.get("replay_previous_probability", -1.0))
    if not 0.0 <= replay < 1.0:
        raise ValueError("replay_previous_probability must be in [0,1)")
    return value


def apply_curriculum_config(env_cfg, value: dict[str, Any], *, evaluation: bool) -> None:
    """把同一份课程定义写入命令、观测和 CurriculumManager 配置。"""

    profiles = []
    for item in value["stages"]:
        profiles.append(
            {
                "min_path_length_m": float(item["path_length_m"][0]),
                "max_path_length_m": float(item["path_length_m"][1]),
                "min_obstacles": int(item["static_obstacles"][0]),
                "max_obstacles": int(item["static_obstacles"][1]),
                "min_dynamic_obstacles": int(item["dynamic_obstacles"][0]),
                "max_dynamic_obstacles": int(item["dynamic_obstacles"][1]),
                "max_curve_offset_m": float(item["max_curve_offset_m"]),
                "goal_yaw_offset_rad": float(item["goal_yaw_offset_deg"]) * 3.141592653589793 / 180.0,
                "randomization_scale": float(item["randomization_scale"]),
            }
        )
    command = env_cfg.commands.navigation
    geometry = value["geometry"]
    command.map_resolution_m = float(geometry["map_resolution_m"])
    command.planning_clearance_m = float(geometry["planning_clearance_m"])
    command.stage_profiles = profiles
    command.curriculum_stage = len(value["stages"]) - 1 if evaluation else int(value["initial_stage"])
    command.replay_previous_probability = (
        0.0 if evaluation else float(value["replay_previous_probability"])
    )
    randomization = value.get("randomization", {})
    lidar = randomization.get("lidar", {})
    localization = randomization.get("localization", {})
    command.lidar_noise_std_m = float(lidar.get("gaussian_std_m", command.lidar_noise_std_m))
    command.lidar_bias_max_m = float(lidar.get("bias_max_m", command.lidar_bias_max_m))
    command.lidar_invalid_ray_probability = float(
        lidar.get("invalid_ray_probability", command.lidar_invalid_ray_probability)
    )
    command.lidar_frame_drop_probability = float(
        lidar.get("whole_frame_drop_probability", command.lidar_frame_drop_probability)
    )
    command.lidar_range_scale_max_error = float(
        lidar.get("range_scale_max_error", command.lidar_range_scale_max_error)
    )
    command.localization_position_std_m = float(
        localization.get("position_std_m", command.localization_position_std_m)
    )
    command.localization_yaw_std_rad = float(
        localization.get("yaw_std_rad", command.localization_yaw_std_rad)
    )
    command.localization_position_drift_mps = float(
        localization.get("slow_position_drift_mps", command.localization_position_drift_mps)
    )
    command.localization_yaw_drift_radps = float(
        localization.get("slow_yaw_drift_radps", command.localization_yaw_drift_radps)
    )
    env_cfg.curriculum.stage.params = {
        "stages": value["stages"],
        "window_episodes": int(value["evaluation_episodes"]),
        "consecutive_passes": int(value["consecutive_passes"]),
    }
