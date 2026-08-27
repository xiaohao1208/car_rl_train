"""带保守对角碰撞处理的栅格搜索基础算法。"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable

import numpy as np

Cell = tuple[int, int]

_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _inside(grid: np.ndarray, cell: Cell) -> bool:
    """判断栅格坐标是否位于地图边界内。"""

    return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]


def valid_neighbors(grid: np.ndarray, cell: Cell) -> Iterable[tuple[Cell, float]]:
    """枚举未占用且不会切角的八邻域及对应移动代价。"""

    for dr, dc, cost in _NEIGHBORS:
        nxt = cell[0] + dr, cell[1] + dc
        if not _inside(grid, nxt) or grid[nxt]:
            continue
        if dr and dc:
            # 对角移动的两个相邻正交栅格必须同时为空，禁止从障碍物角点间穿过。
            if grid[cell[0] + dr, cell[1]] or grid[cell[0], cell[1] + dc]:
                continue
        yield nxt, cost


def astar(grid: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    """返回八连通最短路径；不可达时返回空列表。"""

    occupied = np.asarray(grid, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    if not _inside(occupied, start) or not _inside(occupied, goal):
        raise ValueError("start and goal must be inside the grid")
    if occupied[start] or occupied[goal]:
        return []

    queue: list[tuple[float, float, Cell]] = [(0.0, 0.0, start)]
    parents: dict[Cell, Cell] = {}
    costs: dict[Cell, float] = {start: 0.0}
    visited: set[Cell] = set()
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        if current == goal:
            result = [current]
            while current in parents:
                current = parents[current]
                result.append(current)
            return list(reversed(result))
        visited.add(current)
        for nxt, edge_cost in valid_neighbors(occupied, current):
            candidate = current_cost + edge_cost
            if candidate >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = candidate
            parents[nxt] = current
            heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
            heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
    return []


def inflate_obstacles(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """不依赖 SciPy，使用离散圆盘对二值障碍栅格进行膨胀。"""

    occupied = np.asarray(grid, dtype=bool)
    if radius_cells <= 0:
        return occupied.copy()
    result = occupied.copy()
    rows, cols = np.nonzero(occupied)
    offsets = [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if dr * dr + dc * dc <= radius_cells * radius_cells
    ]
    for dr, dc in offsets:
        shifted_rows, shifted_cols = rows + dr, cols + dc
        valid = (
            (shifted_rows >= 0)
            & (shifted_rows < occupied.shape[0])
            & (shifted_cols >= 0)
            & (shifted_cols < occupied.shape[1])
        )
        result[shifted_rows[valid], shifted_cols[valid]] = True
    return result
