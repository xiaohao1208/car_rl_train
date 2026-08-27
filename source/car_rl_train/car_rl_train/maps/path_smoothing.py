"""折线路径简化、安全平滑及按弧长重采样。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]
Cell = tuple[int, int]


def _supercover(first: Cell, second: Cell) -> list[Cell]:
    """保守枚举两个栅格中心连线所触及的所有单元格。"""

    dr, dc = second[0] - first[0], second[1] - first[1]
    steps = max(abs(dr), abs(dc)) * 4 + 1
    result: list[Cell] = []
    for fraction in np.linspace(0.0, 1.0, max(steps, 2)):
        row = int(round(first[0] + fraction * dr))
        col = int(round(first[1] + fraction * dc))
        cell = row, col
        if not result or result[-1] != cell:
            result.append(cell)
    return result


def line_is_free(grid: np.ndarray, first: Cell, second: Cell) -> bool:
    """判断两格之间的直线是否完全处于自由空间且不切角。"""

    cells = _supercover(first, second)
    for cell in cells:
        if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
            return False
        if grid[cell]:
            return False
    # 显式检查相邻对角栅格，避免视线检测在障碍角点处切角。
    for previous, current in zip(cells, cells[1:]):
        dr, dc = current[0] - previous[0], current[1] - previous[1]
        if abs(dr) == 1 and abs(dc) == 1:
            if grid[previous[0] + dr, previous[1]] or grid[previous[0], previous[1] + dc]:
                return False
    return True


def simplify_path(path: Sequence[Cell], grid: np.ndarray) -> list[Cell]:
    """用最长安全视线段简化离散路径，同时保留首尾点。"""

    if len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not line_is_free(grid, path[anchor], path[candidate]):
            candidate -= 1
        simplified.append(path[candidate])
        anchor = candidate
    return simplified


def smooth_path(
    points: Sequence[Point],
    grid: np.ndarray,
    resolution: float,
    origin: Point = (0.0, 0.0),
    iterations: int = 40,
    weight: float = 0.35,
) -> list[Point]:
    """迭代平滑内部路径点，并拒绝任何进入障碍物的候选位置。"""

    if len(points) <= 2:
        return list(points)
    current = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        updated = current.copy()
        for index in range(1, len(current) - 1):
            candidate = (1.0 - weight) * current[index] + weight * 0.5 * (
                current[index - 1] + current[index + 1]
            )
            row = int((candidate[1] - origin[1]) / resolution)
            col = int((candidate[0] - origin[0]) / resolution)
            if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]) or grid[row, col]:
                continue
            previous = (
                int((updated[index - 1, 1] - origin[1]) / resolution),
                int((updated[index - 1, 0] - origin[0]) / resolution),
            )
            following = (
                int((current[index + 1, 1] - origin[1]) / resolution),
                int((current[index + 1, 0] - origin[0]) / resolution),
            )
            candidate_cell = row, col
            if line_is_free(grid, previous, candidate_cell) and line_is_free(grid, candidate_cell, following):
                updated[index] = candidate
        current = updated
    return [tuple(point) for point in current.tolist()]


def resample_path(points: Sequence[Point], max_spacing: float = 0.05) -> list[Point]:
    """沿折线弧长重采样，保证相邻点间距不超过给定上限。"""

    if not points:
        return []
    if max_spacing <= 0:
        raise ValueError("max_spacing must be positive")
    result = [tuple(points[0])]
    for first, second in zip(points, points[1:]):
        length = math.dist(first, second)
        segments = max(1, int(math.ceil(length / max_spacing)))
        for step in range(1, segments + 1):
            fraction = step / segments
            result.append(
                (
                    first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]),
                )
            )
    return result
