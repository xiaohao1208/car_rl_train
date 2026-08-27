"""[必选] 对全部课程阶段的地图执行确定性、路径和安全净空校验。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source" / "car_rl_train"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from car_rl_train.maps import DEFAULT_STAGE_CONFIGS, generate_navigation_map
from car_rl_train.tasks.local_navigation.geometry import (
    FOOTPRINT_RADIUS_M,
    PATH_SAFETY_MARGIN_M,
    PLANNING_CLEARANCE_M,
)
from car_rl_train.workflow import atomic_write_json


KNOWN_REGRESSION_SEEDS = (293, 412, 1649, 300174, 600090, 700245, 800075, 800248)


def validate_map(
    seed: int, stage: int, *, check_determinism: bool = False
) -> dict[str, float | bool | int]:
    """生成并验证一张地图，返回压力测试统计字段。"""

    config = DEFAULT_STAGE_CONFIGS[stage]
    navigation_map = generate_navigation_map(seed, stage=stage)
    points = np.asarray(navigation_map.path_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise AssertionError(f"seed={seed} stage={stage}: invalid path shape")
    arc_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    tolerance = 1.0e-8
    if not (
        config.min_path_length_m - tolerance
        <= arc_length
        <= config.max_path_length_m + tolerance
    ):
        raise AssertionError(
            f"seed={seed} stage={stage}: arc_length={arc_length}"
        )
    if not config.min_obstacles <= navigation_map.obstacle_count <= config.max_obstacles:
        raise AssertionError(f"seed={seed} stage={stage}: obstacle count")
    rows, cols = navigation_map.inflated.shape
    for x, y in points:
        row = min(rows - 1, max(0, int(y / navigation_map.resolution_m)))
        col = min(cols - 1, max(0, int(x / navigation_map.resolution_m)))
        if navigation_map.inflated[row, col]:
            raise AssertionError(
                f"seed={seed} stage={stage}: path intersects inflated map"
            )
    occupied = np.argwhere(navigation_map.occupied)
    obstacle_xy = np.stack(
        (
            (occupied[:, 1] + 0.5) * navigation_map.resolution_m,
            (occupied[:, 0] + 0.5) * navigation_map.resolution_m,
        ),
        axis=1,
    )
    center_clearance = float(
        np.sqrt(
            np.sum(
                (points[:, None, :] - obstacle_xy[None, :, :]) ** 2,
                axis=2,
            )
        ).min()
    )
    footprint_margin = center_clearance - FOOTPRINT_RADIUS_M
    if center_clearance < PLANNING_CLEARANCE_M - 1.0e-9:
        raise AssertionError(
            f"seed={seed} stage={stage}: center clearance={center_clearance}"
        )
    if footprint_margin < PATH_SAFETY_MARGIN_M - 1.0e-9:
        raise AssertionError(
            f"seed={seed} stage={stage}: footprint margin={footprint_margin}"
        )
    delta = points[-1] - points[-2]
    expected = math.atan2(float(delta[1]), float(delta[0]))
    error = math.atan2(
        math.sin(navigation_map.goal_yaw - expected),
        math.cos(navigation_map.goal_yaw - expected),
    )
    if abs(error) > config.goal_yaw_offset_rad + 1.0e-9:
        raise AssertionError(f"seed={seed} stage={stage}: goal yaw")
    if check_determinism:
        repeated = generate_navigation_map(seed, stage=stage)
        if (
            repeated.path_xy != navigation_map.path_xy
            or not np.array_equal(repeated.occupied, navigation_map.occupied)
            or repeated.goal_yaw != navigation_map.goal_yaw
            or repeated.fallback_used != navigation_map.fallback_used
        ):
            raise AssertionError(f"seed={seed} stage={stage}: non-deterministic map")
    return {
        "arc_length_m": arc_length,
        "attempts": navigation_map.generation_attempts,
        "fallback": navigation_map.fallback_used,
        "center_clearance_m": center_clearance,
        "footprint_margin_m": footprint_margin,
    }


def validate_job(job: tuple[int, int, bool]) -> dict[str, float | bool | int]:
    """供多进程执行器调用的可序列化地图校验入口。"""

    seed, stage, check_determinism = job
    return validate_map(seed, stage, check_determinism=check_determinism)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-per-stage", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args()
    if args.seeds_per_stage <= 0:
        raise ValueError("seeds-per-stage must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    report = {
        "schema_version": 1,
        "seeds_per_stage": args.seeds_per_stage,
        "workers": args.workers,
        "stages": [],
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for stage in range(len(DEFAULT_STAGE_CONFIGS)):
            started = time.monotonic()
            regression_samples = len(KNOWN_REGRESSION_SEEDS) if stage == 0 else 0
            jobs = [
                (args.seed_start + seed, stage, seed == 0)
                for seed in range(args.seeds_per_stage)
            ]
            if stage == 0:
                jobs.extend((seed, stage, True) for seed in KNOWN_REGRESSION_SEEDS)
            values = list(executor.map(validate_job, jobs, chunksize=32))
            report["stages"].append(
                {
                    "stage": stage,
                    "random_samples": args.seeds_per_stage,
                    "regression_samples": regression_samples,
                    "validated_total": len(values),
                    "fallback_count": sum(bool(value["fallback"]) for value in values),
                    "max_attempts": max(int(value["attempts"]) for value in values),
                    "min_center_clearance_m": round(
                        min(float(value["center_clearance_m"]) for value in values), 4
                    ),
                    "min_footprint_margin_m": round(
                        min(float(value["footprint_margin_m"]) for value in values), 4
                    ),
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
            )
            print(json.dumps(report["stages"][-1], ensure_ascii=False), flush=True)
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
