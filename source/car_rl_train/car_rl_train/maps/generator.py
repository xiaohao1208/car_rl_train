"""供仿真环境和轻量测试使用的确定性二维房间生成器。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .path_search import astar, inflate_obstacles
from .path_smoothing import resample_path, simplify_path, smooth_path
from ..tasks.local_navigation.geometry import MAP_RESOLUTION_M, PLANNING_CLEARANCE_M


@dataclass(frozen=True)
class MapConfig:
    """程序化地图的尺寸、膨胀半径和生成重试上限。"""

    width_m: float = 6.0
    height_m: float = 6.0
    resolution_m: float = MAP_RESOLUTION_M
    inflation_radius_m: float = PLANNING_CLEARANCE_M
    min_path_length_m: float = 1.0
    max_path_length_m: float = 4.0
    min_clearance_m: float = PLANNING_CLEARANCE_M
    max_obstacles: int = 20
    max_attempts: int = 100


@dataclass(frozen=True)
class StageMapConfig:
    """单个课程阶段的路线、障碍和目标朝向难度。"""

    min_path_length_m: float
    max_path_length_m: float
    min_obstacles: int
    max_obstacles: int
    min_dynamic_obstacles: int
    max_dynamic_obstacles: int
    max_curve_offset_m: float
    goal_yaw_offset_rad: float
    randomization_scale: float


DEFAULT_STAGE_CONFIGS: tuple[StageMapConfig, ...] = (
    StageMapConfig(0.5, 1.0, 0, 0, 0, 0, 0.15, 0.0, 0.0),
    StageMapConfig(0.6, 1.1, 0, 0, 0, 0, 0.20, math.radians(5.0), 0.02),
    StageMapConfig(0.75, 1.25, 0, 0, 0, 0, 0.27, math.radians(15.0), 0.05),
    StageMapConfig(0.75, 1.5, 0, 0, 0, 0, 0.35, math.radians(30.0), 0.10),
    StageMapConfig(1.0, 2.0, 0, 3, 0, 0, 0.35, math.radians(45.0), 0.25),
    StageMapConfig(1.0, 2.5, 2, 8, 0, 1, 0.35, math.radians(90.0), 0.50),
    StageMapConfig(1.0, 3.5, 4, 14, 0, 2, 0.35, math.radians(150.0), 0.80),
    StageMapConfig(1.0, 4.0, 6, 18, 0, 2, 0.35, math.pi, 1.0),
)


def _curved_free_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    max_offset_m: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """在无障碍阶段生成平滑二次 Bezier 曲线，显式教授路径跟随。"""

    start_array = np.asarray(start, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    direction = goal_array - start_array
    distance = float(np.linalg.norm(direction))
    if max_offset_m <= 0.0 or distance <= 1.0e-9:
        return [start, goal]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64) / distance
    magnitude = float(rng.uniform(0.35 * max_offset_m, max_offset_m))
    offset = magnitude if float(rng.random()) < 0.5 else -magnitude
    control = 0.5 * (start_array + goal_array) + normal * offset
    curve: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, 33):
        point = (1.0 - t) ** 2 * start_array + 2.0 * (1.0 - t) * t * control + t**2 * goal_array
        curve.append((float(point[0]), float(point[1])))
    return curve


@dataclass(frozen=True)
class NavigationMap:
    """一次确定性导航任务生成的地图、起终点及安全路径。"""

    occupied: np.ndarray
    inflated: np.ndarray
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    goal_yaw: float
    path_xy: tuple[tuple[float, float], ...]
    seed: int
    obstacle_count: int
    resolution_m: float
    generation_attempts: int
    fallback_used: bool


def _cell_to_world(cell: tuple[int, int], resolution: float) -> tuple[float, float]:
    """将行列栅格坐标转换为对应单元中心的世界坐标。"""

    return (cell[1] + 0.5) * resolution, (cell[0] + 0.5) * resolution


def _arc_length(points: list[tuple[float, float]]) -> float:
    """返回折线路径弧长。"""

    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def _length_is_valid(length: float, stage_config: StageMapConfig) -> bool:
    """容忍浮点重采样在边界产生的机器精度误差。"""

    tolerance = 1.0e-9
    return (
        stage_config.min_path_length_m - tolerance
        <= length
        <= stage_config.max_path_length_m + tolerance
    )


def _goal_yaw(
    sampled: list[tuple[float, float]],
    stage_config: StageMapConfig,
    rng: np.random.Generator,
) -> float:
    final_delta = np.asarray(sampled[-1]) - np.asarray(sampled[-2])
    path_yaw = math.atan2(float(final_delta[1]), float(final_delta[0]))
    offset = float(stage_config.goal_yaw_offset_rad)
    yaw = path_yaw if offset == 0.0 else path_yaw + float(rng.uniform(-offset, offset))
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _path_is_free(
    sampled: list[tuple[float, float]],
    inflated: np.ndarray,
    resolution: float,
) -> bool:
    """按仿真占据栅格索引规则复验所有重采样点。"""

    rows, cols = inflated.shape
    for x, y in sampled:
        row = min(rows - 1, max(0, int(y / resolution)))
        col = min(cols - 1, max(0, int(x / resolution)))
        if inflated[row, col]:
            return False
    return True


def _fallback_navigation_map(
    seed: int,
    config: MapConfig,
    stage_config: StageMapConfig,
) -> NavigationMap:
    """构造始终可行的直线路径和远离路径的确定性障碍布局。"""

    rng = np.random.default_rng(seed ^ 0x6A09E667)
    rows = int(round(config.height_m / config.resolution_m))
    cols = int(round(config.width_m / config.resolution_m))
    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    minimum = max(0, int(stage_config.min_obstacles))
    maximum = min(config.max_obstacles, int(stage_config.max_obstacles))
    obstacle_count = (
        minimum
        if minimum == maximum
        else int(rng.integers(minimum, maximum + 1))
    )

    # 两条远离中心通道的障碍带最多容纳18个独立小障碍，覆盖当前课程配置。
    x_positions = np.linspace(0.75, config.width_m - 0.75, 9)
    centers = [
        (0.75, float(x)) for x in x_positions
    ] + [
        (config.height_m - 0.75, float(x)) for x in x_positions
    ]
    if obstacle_count > len(centers):
        raise ValueError("fallback layout cannot place requested obstacle count")
    order = rng.permutation(len(centers))[:obstacle_count]
    for index, center_index in enumerate(order):
        y, x = centers[int(center_index)]
        cell = (
            int(round(y / config.resolution_m)),
            int(round(x / config.resolution_m)),
        )
        if index % 2:
            _draw_rect(occupied, cell, (2, 2))
        else:
            _draw_circle(occupied, cell, 2)

    clearance_cells = int(math.ceil(config.min_clearance_m / config.resolution_m))
    inflation_cells = int(math.ceil(config.inflation_radius_m / config.resolution_m))
    inflated = inflate_obstacles(
        occupied, max(clearance_cells, inflation_cells)
    )
    available = config.width_m - 2.0
    if stage_config.min_path_length_m > available:
        raise ValueError("fallback layout cannot satisfy minimum path length")
    target_length = min(
        available,
        0.5
        * (
            stage_config.min_path_length_m
            + stage_config.max_path_length_m
        ),
    )
    start = (1.0, 0.5 * config.height_m)
    goal = (1.0 + target_length, 0.5 * config.height_m)
    sampled = resample_path([start, goal], config.resolution_m)
    if not _length_is_valid(_arc_length(sampled), stage_config):
        raise ValueError("fallback layout violates path length contract")
    if not _path_is_free(sampled, inflated, config.resolution_m):
        raise ValueError("fallback path intersects inflated obstacles")
    return NavigationMap(
        occupied=occupied,
        inflated=inflated,
        start_xy=sampled[0],
        goal_xy=sampled[-1],
        goal_yaw=_goal_yaw(sampled, stage_config, rng),
        path_xy=tuple(sampled),
        seed=seed,
        obstacle_count=obstacle_count,
        resolution_m=config.resolution_m,
        generation_attempts=config.max_attempts,
        fallback_used=True,
    )


def _draw_rect(grid: np.ndarray, center: tuple[int, int], half: tuple[int, int]) -> None:
    """在栅格内绘制轴对齐矩形障碍物。"""

    r0, r1 = max(1, center[0] - half[0]), min(grid.shape[0] - 1, center[0] + half[0] + 1)
    c0, c1 = max(1, center[1] - half[1]), min(grid.shape[1] - 1, center[1] + half[1] + 1)
    grid[r0:r1, c0:c1] = True


def _draw_circle(grid: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """在栅格内绘制离散圆形障碍物。"""

    rows, cols = np.ogrid[: grid.shape[0], : grid.shape[1]]
    grid[(rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius * radius] = True


def _build_occupancy(
    rng: np.random.Generator, config: MapConfig, stage_config: StageMapConfig
) -> tuple[np.ndarray, int]:
    """按照课程阶段生成矩形、圆形和 L 形静态障碍。"""

    rows = int(round(config.height_m / config.resolution_m))
    cols = int(round(config.width_m / config.resolution_m))
    grid = np.zeros((rows, cols), dtype=bool)
    grid[[0, -1], :] = True
    grid[:, [0, -1]] = True
    minimum = max(0, int(stage_config.min_obstacles))
    maximum = min(config.max_obstacles, int(stage_config.max_obstacles))
    if maximum < minimum:
        raise ValueError("stage obstacle range is invalid")
    obstacle_count = minimum if minimum == maximum else int(rng.integers(minimum, maximum + 1))
    for _ in range(obstacle_count):
        center = int(rng.integers(8, rows - 8)), int(rng.integers(8, cols - 8))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            _draw_rect(grid, center, (int(rng.integers(2, 6)), int(rng.integers(2, 8))))
        elif kind == 1:
            _draw_circle(grid, center, int(rng.integers(2, 6)))
        else:
            first_half = (int(rng.integers(2, 6)), int(rng.integers(1, 3)))
            second_half = (int(rng.integers(1, 3)), int(rng.integers(2, 6)))
            _draw_rect(grid, center, first_half)
            offset = center[0] + first_half[0] - second_half[0], center[1] + second_half[1]
            _draw_rect(grid, offset, second_half)
    return grid, obstacle_count


def generate_navigation_map(
    seed: int,
    stage: int = 5,
    config: MapConfig = MapConfig(),
    stage_config: StageMapConfig | None = None,
) -> NavigationMap:
    """生成有效地图；随机重试耗尽时返回确定、安全的保底地图。"""

    if not 0 <= stage < len(DEFAULT_STAGE_CONFIGS):
        raise ValueError(
            f"curriculum stage must be in [0, {len(DEFAULT_STAGE_CONFIGS) - 1}]"
        )
    stage_config = stage_config or DEFAULT_STAGE_CONFIGS[stage]
    if (
        stage_config.min_path_length_m <= 0.0
        or stage_config.max_path_length_m < stage_config.min_path_length_m
        or not 0.0 <= stage_config.goal_yaw_offset_rad <= math.pi
        or not 0.0 <= stage_config.randomization_scale <= 1.0
        or stage_config.min_dynamic_obstacles < 0
        or stage_config.max_dynamic_obstacles < stage_config.min_dynamic_obstacles
        or not 0.0 <= stage_config.max_curve_offset_m <= 1.0
    ):
        raise ValueError("invalid stage map configuration")
    rng = np.random.default_rng(seed)
    clearance_cells = int(math.ceil(config.min_clearance_m / config.resolution_m))
    inflation_cells = int(math.ceil(config.inflation_radius_m / config.resolution_m))
    for attempt in range(1, config.max_attempts + 1):
        occupied, obstacle_count = _build_occupancy(rng, config, stage_config)
        inflated = inflate_obstacles(occupied, max(clearance_cells, inflation_cells))
        free = np.argwhere(~inflated)
        if len(free) < 2:
            continue
        chosen = free[rng.choice(len(free), size=2, replace=False)]
        start, goal = tuple(chosen[0]), tuple(chosen[1])
        distance = math.dist(_cell_to_world(start, config.resolution_m), _cell_to_world(goal, config.resolution_m))
        if not stage_config.min_path_length_m <= distance <= stage_config.max_path_length_m:
            continue
        if obstacle_count == 0:
            smoothed = _curved_free_path(
                _cell_to_world(start, config.resolution_m),
                _cell_to_world(goal, config.resolution_m),
                stage_config.max_curve_offset_m,
                rng,
            )
        else:
            cells = astar(inflated, start, goal)
            if not cells:
                continue
            simplified_cells = simplify_path(cells, inflated)
            world_points = [
                _cell_to_world(cell, config.resolution_m)
                for cell in simplified_cells
            ]
            smoothed = smooth_path(
                world_points, inflated, config.resolution_m
            )
        sampled = resample_path(smoothed, config.resolution_m)
        if len(sampled) < 2:
            continue
        arc_length = _arc_length(sampled)
        if not _length_is_valid(arc_length, stage_config):
            continue
        if not _path_is_free(sampled, inflated, config.resolution_m):
            continue
        return NavigationMap(
            occupied=occupied,
            inflated=inflated,
            start_xy=sampled[0],
            goal_xy=sampled[-1],
            goal_yaw=_goal_yaw(sampled, stage_config, rng),
            path_xy=tuple(sampled),
            seed=seed,
            obstacle_count=obstacle_count,
            resolution_m=config.resolution_m,
            generation_attempts=attempt,
            fallback_used=False,
        )
    return _fallback_navigation_map(seed, config, stage_config)
