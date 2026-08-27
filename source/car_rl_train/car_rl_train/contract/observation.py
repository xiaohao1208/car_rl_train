"""严格复刻 C++ 控制器观测契约的 Python 实现。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

OBSERVATION_SIZE = 86
SCAN_RAY_COUNT = 72
LOOKAHEAD_DISTANCES = (0.3, 0.6, 1.0)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """统一读取字典字段或对象属性，便于兼容 ROS 消息和测试字典。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _yaw(pose: Any) -> float:
    """从平面位姿四元数中提取规范化 yaw 角。"""

    orientation = _field(pose, "orientation")
    if orientation is None:
        x = float(_field(pose, "qx", 0.0))
        y = float(_field(pose, "qy", 0.0))
        z = float(_field(pose, "qz", 0.0))
        w = float(_field(pose, "qw", 1.0))
    else:
        x = float(_field(orientation, "x", 0.0))
        y = float(_field(orientation, "y", 0.0))
        z = float(_field(orientation, "z", 0.0))
        w = float(_field(orientation, "w", 1.0))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _xy(pose: Any) -> tuple[float, float]:
    """从位姿或位置对象中读取二维坐标。"""

    position = _field(pose, "position", pose)
    return float(_field(position, "x")), float(_field(position, "y"))


def _world_to_base(point: tuple[float, float], robot_pose: Any) -> tuple[float, float]:
    """把世界坐标点变换到机器人底盘坐标系。"""

    rx, ry = _xy(robot_pose)
    yaw = _yaw(robot_pose)
    dx, dy = point[0] - rx, point[1] - ry
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def normalize_scan(
    scan: Any,
    range_min: float = 0.05,
    range_max: float = 6.0,
) -> np.ndarray:
    """将任意分辨率 LaserScan 线性重采样到契约规定的 72 个角度。"""

    ranges = np.asarray(_field(scan, "ranges"), dtype=np.float32)
    angle_min = np.float32(_field(scan, "angle_min"))
    angle_increment = np.float32(_field(scan, "angle_increment"))
    range_min_f = np.float32(range_min)
    range_max_f = np.float32(range_max)
    if ranges.size == 0 or angle_increment == 0 or range_max_f <= range_min_f:
        raise ValueError("invalid LaserScan parameters")

    result = np.empty(SCAN_RAY_COUNT, dtype=np.float32)
    for ray in range(SCAN_RAY_COUNT):
        target_angle = -math.pi + 2.0 * math.pi * ray / SCAN_RAY_COUNT
        source_index = (target_angle - float(angle_min)) / float(angle_increment)
        distance = range_max_f
        if 0.0 <= source_index <= ranges.size - 1:
            lower = int(math.floor(source_index))
            upper = min(lower + 1, ranges.size - 1)
            weight = source_index - lower
            lower_value, upper_value = ranges[lower], ranges[upper]
            lower_valid = np.isfinite(lower_value) and lower_value >= range_min_f
            upper_valid = np.isfinite(upper_value) and upper_value >= range_min_f
            if lower_valid and upper_valid:
                distance = np.float32(
                    float(lower_value) * (1.0 - weight) + float(upper_value) * weight
                )
            elif lower_valid:
                distance = lower_value
            elif upper_valid:
                distance = upper_value
        distance = np.float32(np.clip(distance, range_min_f, range_max_f))
        result[ray] = np.float32((distance - range_min_f) / (range_max_f - range_min_f))
    return result


def sample_path_points(path: Sequence[Any], robot_pose: Any) -> np.ndarray:
    """投影到完整路径折线，并按固定前向弧长采样三个路径点。"""

    if not path:
        raise ValueError("global path is empty")
    nearest_segment, nearest_ratio = 0, 0.0
    nearest_distance = math.inf
    robot_x, robot_y = _xy(robot_pose)
    if len(path) > 1:
        for index in range(len(path) - 1):
            first, second = _xy(path[index]), _xy(path[index + 1])
            dx, dy = second[0] - first[0], second[1] - first[1]
            length_squared = dx * dx + dy * dy
            ratio = 0.0 if length_squared <= 1.0e-12 else np.clip(
                ((robot_x - first[0]) * dx + (robot_y - first[1]) * dy) / length_squared,
                0.0,
                1.0,
            )
            px, py = first[0] + ratio * dx, first[1] + ratio * dy
            candidate = math.hypot(px - robot_x, py - robot_y)
            if candidate < nearest_distance:
                nearest_distance = candidate
                nearest_segment = index
                nearest_ratio = float(ratio)

    sampled = np.empty(6, dtype=np.float32)
    path_end = _xy(path[-1])
    for target_index, lookahead in enumerate(LOOKAHEAD_DISTANCES):
        remaining = lookahead
        sample_x, sample_y = path_end
        for index in range(nearest_segment, len(path) - 1):
            first, second = _xy(path[index]), _xy(path[index + 1])
            segment_length = math.dist(first, second)
            if segment_length <= 1.0e-9:
                continue
            start_ratio = nearest_ratio if index == nearest_segment else 0.0
            available = segment_length * (1.0 - start_ratio)
            if remaining <= available:
                ratio = start_ratio + remaining / segment_length
                sample_x = first[0] + ratio * (second[0] - first[0])
                sample_y = first[1] + ratio * (second[1] - first[1])
                break
            remaining -= available
        relative = _world_to_base((sample_x, sample_y), robot_pose)
        sampled[target_index * 2] = np.float32(np.clip(relative[0], -1.5, 1.5) / 1.5)
        sampled[target_index * 2 + 1] = np.float32(np.clip(relative[1], -1.5, 1.5) / 1.5)
    return sampled


def build_controller_observation(
    scan: Any,
    path: Sequence[Any],
    robot_pose: Any,
    velocity: Any,
    previous_action: Sequence[float],
) -> np.ndarray:
    """按固定字段顺序构造严格的 86 维 float32 控制器观测。"""

    if not path:
        raise ValueError("global path is empty")
    observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    observation[:72] = normalize_scan(scan)
    observation[72:78] = sample_path_points(path, robot_pose)
    goal_relative = _world_to_base(_xy(path[-1]), robot_pose)
    yaw_error = math.atan2(
        math.sin(_yaw(path[-1]) - _yaw(robot_pose)),
        math.cos(_yaw(path[-1]) - _yaw(robot_pose)),
    )
    observation[78] = np.float32(np.clip(goal_relative[0], -3.0, 3.0) / 3.0)
    observation[79] = np.float32(np.clip(goal_relative[1], -3.0, 3.0) / 3.0)
    observation[80] = np.float32(math.sin(yaw_error))
    observation[81] = np.float32(math.cos(yaw_error))
    linear = float(_field(velocity, "linear_x", _field(velocity, "linear", 0.0)))
    angular = float(_field(velocity, "angular_z", _field(velocity, "angular", 0.0)))
    observation[82] = np.float32(np.clip(linear, -0.25, 0.25) / 0.25)
    observation[83] = np.float32(np.clip(angular, -1.0, 1.0))
    previous = np.asarray(previous_action, dtype=np.float32)
    if previous.shape != (2,):
        raise ValueError("previous_action must have shape [2]")
    observation[84:86] = np.clip(previous, -1.0, 1.0)
    if not np.isfinite(observation).all():
        raise ValueError("controller observation contains non-finite values")
    return observation


def build_from_golden_case(case: Mapping[str, Any]) -> np.ndarray:
    """从跨语言黄金向量样例中提取输入并复算观测。"""

    return build_controller_observation(
        case["scan"],
        case["path"],
        case["robot_pose"],
        case["velocity"],
        case["previous_action"],
    )
