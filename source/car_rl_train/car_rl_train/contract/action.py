"""归一化控制动作与底盘速度之间的映射。"""

from __future__ import annotations

import numpy as np

MAX_LINEAR_SPEED_MPS = 0.10
MAX_ANGULAR_SPEED_RADPS = 1.047197551


def action_to_twist(
    action: np.ndarray | list[float] | tuple[float, float],
    max_linear_speed: float = MAX_LINEAR_SPEED_MPS,
    max_angular_speed: float = MAX_ANGULAR_SPEED_RADPS,
) -> np.ndarray:
    """将归一化动作映射为 ``[linear_x, angular_z]``。

    返回值使用 float64，因为 ROS ``Twist`` 成员是双精度浮点数。
    """

    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1:] != (2,):
        raise ValueError(f"action must end in dimension 2, got {values.shape}")
    clipped = np.clip(values.astype(np.float64), -1.0, 1.0)
    result = np.empty_like(clipped, dtype=np.float64)
    result[..., 0] = 0.5 * float(max_linear_speed) * (clipped[..., 0] + 1.0)
    result[..., 1] = float(max_angular_speed) * clipped[..., 1]
    return result


def twist_to_wheel_rates(
    twist: np.ndarray,
    wheel_radius_m: float = 0.032,
    wheel_separation_m: float = 0.175,
) -> np.ndarray:
    """将底盘线速度、角速度换算为左右轮角速度。"""

    values = np.asarray(twist, dtype=np.float64)
    if values.shape[-1:] != (2,):
        raise ValueError("twist must end in [linear_x, angular_z]")
    linear, angular = values[..., 0], values[..., 1]
    left = (linear - angular * wheel_separation_m / 2.0) / wheel_radius_m
    right = (linear + angular * wheel_separation_m / 2.0) / wheel_radius_m
    return np.stack((left, right), axis=-1)
