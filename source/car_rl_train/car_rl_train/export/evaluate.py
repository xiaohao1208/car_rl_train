"""评估指标汇总与确定性候选模型排序。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    """一个固定测试种子回合的原始评估结果。"""

    success: bool
    collision: bool
    static_collision: bool
    dynamic_collision: bool
    timeout: bool
    stall: bool
    path_deviation: bool
    invalid_input: bool
    termination_reason: str
    path_length_m: float
    cross_track_errors_m: tuple[float, ...]
    goal_position_error_m: float
    goal_yaw_error_rad: float
    episode_time_sec: float
    action_deltas: tuple[float, ...]
    minimum_clearance_m: float
    seed: int


def summarize_episodes(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """汇总单次运行的成功、安全、精度和平滑性指标。"""

    if not episodes:
        raise ValueError("at least one episode is required")
    cross_track = np.concatenate(
        [np.asarray(item.cross_track_errors_m, dtype=np.float64) for item in episodes]
    )
    action_deltas = np.concatenate(
        [np.asarray(item.action_deltas, dtype=np.float64) for item in episodes]
    )
    count = len(episodes)
    reached = [item for item in episodes if item.success]
    # “终点精度”只描述真正到达终点的样本。失败回合已经由成功率及各类
    # 失败率独立约束；再把它们混入终点精度会对同一次失败重复惩罚。
    # 全回合误差仍保留为诊断字段。全失败时使用全体样本，保证指标有限。
    goal_accuracy_episodes = reached or episodes
    return {
        "schema_version": 4,
        "episodes": count,
        "success_rate": sum(item.success for item in episodes) / count,
        "collision_rate": sum(item.collision for item in episodes) / count,
        "static_collision_rate": sum(item.static_collision for item in episodes) / count,
        "dynamic_collision_rate": sum(item.dynamic_collision for item in episodes) / count,
        "timeout_rate": sum(item.timeout for item in episodes) / count,
        "stall_rate": sum(item.stall for item in episodes) / count,
        "path_deviation_rate": sum(item.path_deviation for item in episodes) / count,
        "invalid_input_rate": sum(item.invalid_input for item in episodes) / count,
        "mean_path_length_m": mean(item.path_length_m for item in episodes),
        "mean_cross_track_error_m": float(np.mean(cross_track)),
        "p95_cross_track_error_m": float(np.percentile(cross_track, 95)),
        "goal_accuracy_episode_count": len(reached),
        "mean_goal_position_error_m": mean(
            item.goal_position_error_m for item in goal_accuracy_episodes
        ),
        "mean_goal_yaw_error_rad": mean(
            item.goal_yaw_error_rad for item in goal_accuracy_episodes
        ),
        "all_episode_mean_goal_position_error_m": mean(
            item.goal_position_error_m for item in episodes
        ),
        "all_episode_mean_goal_yaw_error_rad": mean(
            item.goal_yaw_error_rad for item in episodes
        ),
        "mean_episode_time_sec": mean(item.episode_time_sec for item in episodes),
        "mean_action_delta": float(np.mean(action_deltas)),
        "minimum_clearance_m": min(item.minimum_clearance_m for item in episodes),
        "test_seed_range": [min(item.seed for item in episodes), max(item.seed for item in episodes)],
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """计算多个随机种子运行中各数值指标的均值与标准差。"""

    if not runs:
        raise ValueError("at least one run is required")
    metric_names = [
        key
        for key, value in runs[0].items()
        if isinstance(value, (int, float)) and key not in {"schema_version", "episodes"}
    ]
    aggregate: dict[str, Any] = {"schema_version": 1, "run_count": len(runs)}
    for name in metric_names:
        values = [float(run[name]) for run in runs]
        aggregate[name] = {"mean": mean(values), "std": pstdev(values)}
    return aggregate


def candidate_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    """按安全优先、成功率次之的固定规则生成候选排序键。"""

    return (
        float(metrics["collision_rate"]),
        -float(metrics["success_rate"]),
        float(metrics["p95_cross_track_error_m"]),
        float(metrics["mean_action_delta"]),
    )
