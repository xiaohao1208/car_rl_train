"""实车标定结果的严格导入、解析和 Isaac 环境配置。"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
METRICS = (
    "motor_time_constant_sec", "action_delay_sec", "left_motor_gain",
    "right_motor_gain", "linear_accel_mps2", "linear_decel_mps2",
    "angular_accel_radps2", "angular_decel_radps2",
)
REQUIRED_COLUMNS = {
    "timestamp_sec", "command_angular_radps",
    "left_target_radps", "right_target_radps",
    "left_feedback_radps", "right_feedback_radps", "phase_id", "segment",
    "trial_id", "request_timestamp_sec", "left_feedback_mps",
    "right_feedback_mps", "imu_bias_corrected_radps",
}
EXPECTED_LEVELS = (
    "A2-+0.030", "A2-+0.050", "A2-+0.080", "A2-+0.100",
    "A3--0.523599", "A3-+0.523599", "A3--0.785398",
    "A3-+0.785398", "A3--1.047198", "A3-+1.047198",
)
MASS_RELATIVE_RANGE = (0.90, 1.10)
FRICTION_RELATIVE_RANGE = (0.80, 1.20)
DYNAMIC_FRICTION_RATIO = 0.80
CASTER_FRICTION = 0.05


class CalibrationError(ValueError):
    """标定文件、质量门禁或 checkpoint 一致性不满足契约。"""


def sha256_file(path: Path) -> str:
    """流式计算文件摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"{path} 必须包含 YAML mapping")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{label} 不是数值") from error
    if not math.isfinite(number):
        raise CalibrationError(f"{label} 不是有限数")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise CalibrationError(f"{label} 必须为正数")
    return number


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise CalibrationError("标定指标没有有限样本")
    return {
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "count": int(len(array)),
    }


def _crossing(time_values, values, threshold, direction):
    hits = values >= threshold if direction > 0 else values <= threshold
    for index in range(max(0, len(hits) - 1)):
        if bool(hits[index]) and bool(hits[index + 1]):
            return float(time_values[index])
    return None


def _slope(time_values, values, start, end, positive):
    magnitude = abs(end - start)
    if magnitude <= 1.0e-9:
        return None
    progress = (values - start) * np.sign(end - start)
    selected = (progress >= 0.10 * magnitude) & (progress <= 0.90 * magnitude)
    if np.count_nonzero(selected) < 3:
        return None
    filtered = np.asarray(values, dtype=np.float64).copy()
    if len(filtered) >= 3:
        filtered[1:-1] = np.median(
            np.stack((filtered[:-2], filtered[1:-1], filtered[2:])), axis=0
        )
    derivative = np.gradient(filtered, time_values)
    candidates = derivative[selected]
    candidates = candidates[candidates > 0] if positive else -candidates[candidates < 0]
    return float(np.percentile(candidates, 95)) if len(candidates) else None


def _encoder_angular_series(hold, rows):
    """由命令/目标轮速比例恢复编码器轮速差对应的车体角速度。"""

    command = np.asarray([
        float(row["command_angular_radps"]) for row in hold
    ])
    target_delta = np.asarray([
        float(row["right_target_radps"])
        - float(row["left_target_radps"])
        for row in hold
    ])
    valid = (
        np.isfinite(command)
        & np.isfinite(target_delta)
        & (np.abs(command) > 1.0e-6)
        & (np.abs(target_delta) > 1.0e-6)
    )
    if np.count_nonzero(valid) < 3:
        return None
    scale = float(np.median(np.abs(command[valid] / target_delta[valid])))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale * np.asarray([
        float(row["right_feedback_radps"])
        - float(row["left_feedback_radps"])
        for row in rows
    ])


def _trial_metrics(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[float]] | None, str]:
    rows = sorted(rows, key=lambda row: float(row["timestamp_sec"]))
    baseline = [row for row in rows if row["segment"] == "baseline"]
    hold = [row for row in rows if row["segment"] == "hold"]
    recovery = [row for row in rows if row["segment"] == "recovery"]
    if not baseline or not hold or not recovery:
        return None, "缺少 baseline/hold/recovery 分段"
    hold_time = np.asarray([float(row["timestamp_sec"]) for row in hold])
    recovery_time = np.asarray([float(row["timestamp_sec"]) for row in recovery])
    if hold_time[-1] - hold_time[0] < 2.5:
        return None, "hold 有效时长不足 2.5s"
    request_time = float(hold[0]["request_timestamp_sec"])
    result: dict[str, list[float]] = defaultdict(list)
    for side in ("left", "right"):
        feedback_key = f"{side}_feedback_radps"
        target_key = f"{side}_target_radps"
        baseline_tail = [
            row for row in baseline
            if float(row["timestamp_sec"]) >= float(baseline[-1]["timestamp_sec"]) - 0.5
        ]
        steady = [
            row for row in hold
            if float(row["timestamp_sec"]) >= hold_time[-1] - 0.8
        ]
        initial = float(np.median([float(row[feedback_key]) for row in baseline_tail]))
        final = float(np.median([float(row[feedback_key]) for row in steady]))
        target = float(np.median([float(row[target_key]) for row in steady]))
        change = final - initial
        if abs(change) < 0.2 or abs(target) < 0.2:
            return None, f"{side} 轮响应幅度不足"
        values = np.asarray([float(row[feedback_key]) for row in hold])
        direction = 1 if change > 0 else -1
        t5 = _crossing(hold_time, values, initial + 0.05 * change, direction)
        t63 = _crossing(hold_time, values, initial + 0.632 * change, direction)
        if t5 is None or t63 is None or t63 <= t5 or t5 < request_time:
            return None, f"{side} 轮响应交点无效"
        result["action_delay_sec"].append(t5 - request_time)
        result["motor_time_constant_sec"].append((t63 - t5) / 0.948707)
        result[f"{side}_motor_gain"].append(final / target)

    linear_hold = np.asarray([
        0.5 * (float(row["left_feedback_mps"]) + float(row["right_feedback_mps"]))
        for row in hold
    ])
    linear_recovery = np.asarray([
        0.5 * (float(row["left_feedback_mps"]) + float(row["right_feedback_mps"]))
        for row in recovery
    ])
    if hold[0]["phase_id"] == "A2":
        acceleration = _slope(
            hold_time, linear_hold, linear_hold[0], np.median(linear_hold[-5:]), True
        )
        deceleration = _slope(
            recovery_time, linear_recovery, linear_recovery[0],
            np.median(linear_recovery[-5:]), False,
        )
        if acceleration is None or deceleration is None:
            return None, "线加减速度区间无效"
        result["linear_accel_mps2"].append(acceleration)
        result["linear_decel_mps2"].append(deceleration)
    else:
        angular_hold = _encoder_angular_series(hold, hold)
        angular_recovery = _encoder_angular_series(hold, recovery)
        if angular_hold is None or angular_recovery is None:
            return None, "编码器角速度比例无效"
        direction = 1.0 if np.median(angular_hold[-5:]) >= 0.0 else -1.0
        acceleration = _slope(
            hold_time, direction * angular_hold, direction * angular_hold[0],
            direction * np.median(angular_hold[-5:]), True,
        )
        deceleration = _slope(
            recovery_time, direction * angular_recovery,
            direction * angular_recovery[0],
            direction * np.median(angular_recovery[-5:]), False,
        )
        if acceleration is None or deceleration is None:
            return None, "角加减速度区间无效"
        result["angular_accel_radps2"].append(acceleration)
        result["angular_decel_radps2"].append(deceleration)
    return dict(result), ""


def analyze_csv(path: Path) -> tuple[dict[str, dict[str, float | int]], dict[str, int]]:
    """从原始 trial 重新估算分布，禁止只信任 YAML 中的汇总值。"""

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise CalibrationError("calibration_raw.csv 为空")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise CalibrationError(f"CSV 缺少列: {sorted(missing)}")
    timestamps = np.asarray([
        _finite(row["timestamp_sec"], "timestamp_sec") for row in rows
    ])
    if np.any(np.diff(timestamps) <= 0.0):
        raise CalibrationError("timestamp_sec 必须严格递增")
    if not {"A0", "A1", "A2", "A3"}.issubset({row["phase_id"] for row in rows}):
        raise CalibrationError("CSV 不包含完整 A0～A3")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for column in REQUIRED_COLUMNS - {"phase_id", "segment", "trial_id"}:
            _finite(row[column], column)
        if row["phase_id"] in {"A2", "A3"} and row["trial_id"]:
            grouped[row["trial_id"]].append(row)
    metrics: dict[str, list[float]] = defaultdict(list)
    levels: dict[str, int] = defaultdict(int)
    rejected: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for trial_id, trial_rows in grouped.items():
        level = trial_id.rsplit("-", 1)[0]
        trial, reason = _trial_metrics(trial_rows)
        if trial is None:
            rejected[level][reason] += 1
            continue
        levels[level] += 1
        for key, values in trial.items():
            metrics[key].extend(values)
    deficient = [level for level in EXPECTED_LEVELS if levels.get(level, 0) < 4]
    if deficient:
        details = []
        for level in deficient:
            reasons = "、".join(
                f"{reason}×{count}"
                for reason, count in sorted(rejected[level].items())
            ) or "未采集到 trial"
            details.append(
                f"{level}={levels.get(level, 0)}/4（{reasons}）"
            )
        raise CalibrationError("有效 trial 不足: " + "; ".join(details))
    distributions = {name: _percentiles(metrics[name]) for name in METRICS}
    invalid = [
        name for name, value in distributions.items()
        if int(value["count"]) < 8 or float(value["p50"]) <= 0.0
    ]
    if invalid:
        raise CalibrationError(f"指标样本不足或非正数: {invalid}")
    return distributions, dict(levels)


def _resolved_payload(run_id: str, yaml_path: Path, csv_path: Path) -> dict[str, Any]:
    calibration = _mapping(yaml_path)
    if calibration.get("schema_version") != 1:
        raise CalibrationError("只支持 calibration schema_version=1")
    if calibration.get("total_mass_kg") is not None:
        _positive(calibration["total_mass_kg"], "total_mass_kg")
    if calibration.get("wheel_friction_center") is not None:
        _positive(
            calibration["wheel_friction_center"], "wheel_friction_center"
        )
    csv_hash = sha256_file(csv_path)
    if calibration.get("source_log_sha256") != csv_hash:
        raise CalibrationError("YAML source_log_sha256 与 CSV 不一致")
    distributions, levels = analyze_csv(csv_path)
    for name in METRICS:
        yaml_value = _finite(calibration.get(name), name)
        recomputed = float(distributions[name]["p50"])
        if not math.isclose(yaml_value, recomputed, rel_tol=1.0e-6, abs_tol=1.0e-9):
            raise CalibrationError(f"{name} 与 CSV 重算中位数不一致")
    policy_hz = 15.0
    delay = distributions["action_delay_sec"]
    quantized = {
        "policy_hz": policy_hz,
        "p10_steps": max(0, math.floor(float(delay["p10"]) * policy_hz)),
        "p50_steps": max(0, round(float(delay["p50"]) * policy_hz)),
        "p90_steps": max(0, math.ceil(float(delay["p90"]) * policy_hz)),
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "measured_at": calibration.get("measured_at"),
        "files": {
            "calibration_yaml_sha256": sha256_file(yaml_path),
            "calibration_csv_sha256": csv_hash,
        },
        "calibration": {name: float(calibration[name]) for name in METRICS},
        "manual_fields": {
            "total_mass_kg": calibration.get("total_mass_kg"),
            "wheel_friction_center": calibration.get("wheel_friction_center"),
        },
        "distributions": distributions,
        "valid_trials": levels,
        "delay_quantization": quantized,
    }


def load_calibration(value: str | Path | None) -> dict[str, Any] | None:
    """从固定 calibration 目录或显式目录读取并实时重算标定。"""

    if value in (None, "", "null"):
        return None
    raw = Path(str(value)).expanduser()
    candidate = raw if raw.is_absolute() else ROOT / raw
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    yaml_path = candidate / "calibration.yaml"
    csv_path = candidate / "calibration_raw.csv"
    if not yaml_path.is_file() or not csv_path.is_file():
        raise CalibrationError(
            "标定目录必须同时包含 calibration.yaml 和 calibration_raw.csv"
        )
    return _resolved_payload(candidate.name, yaml_path, csv_path)


def select_calibration(
    explicit: str | Path | None,
    configured: str | Path | None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """按 CLI、配置、checkpoint 顺序选择，并拒绝 resume 时偷换标定。"""

    requested = load_calibration(explicit if explicit is not None else configured)
    embedded = checkpoint.get("calibration") if checkpoint else None
    if embedded and requested:
        embedded_identity = embedded.get("files")
        requested_identity = requested.get("files")
        if embedded_identity != requested_identity:
            raise CalibrationError("显式标定与 checkpoint 内嵌标定不一致")
    return requested or embedded


def apply_calibration(
    env_cfg,
    calibration: dict[str, Any] | None,
    *,
    training: bool,
) -> dict[str, Any]:
    """把实测动力学、总质量和接触摩擦写入环境配置。"""

    if calibration is None:
        mass_range = tuple(env_cfg.events.total_mass.params["total_mass_range_kg"])
        static_range = tuple(
            env_cfg.events.wheel_material.params["static_friction_range"]
        )
        dynamic_range = tuple(
            env_cfg.events.wheel_material.params["dynamic_friction_range"]
        )
        return {
            "source": "bootstrap",
            "training_randomization": bool(training),
            "total_mass_range_kg": list(mass_range),
            "wheel_static_friction_range": list(static_range),
            "wheel_dynamic_friction_range": list(dynamic_range),
            "caster_friction": CASTER_FRICTION,
        }
    manual = calibration.get("manual_fields")
    if not isinstance(manual, dict):
        raise CalibrationError("resolved calibration 缺少 manual_fields")
    missing = [
        name for name in ("total_mass_kg", "wheel_friction_center")
        if manual.get(name) is None
    ]
    if missing:
        raise CalibrationError(
            "实车物理标定不完整，缺少: " + ", ".join(missing)
        )
    mass = _positive(manual["total_mass_kg"], "total_mass_kg")
    static_friction = _positive(
        manual["wheel_friction_center"], "wheel_friction_center"
    )

    drive = env_cfg.actions.drive
    distributions = calibration["distributions"]
    for name in METRICS:
        if name == "motor_time_constant_sec":
            prefix = "motor_time_constant_sec"
        elif name in {"left_motor_gain", "right_motor_gain"}:
            prefix = name
        elif name == "linear_accel_mps2":
            prefix = "max_linear_accel_mps2"
        elif name == "linear_decel_mps2":
            prefix = "max_linear_decel_mps2"
        elif name == "angular_accel_radps2":
            prefix = "max_angular_accel_radps2"
        elif name == "angular_decel_radps2":
            prefix = "max_angular_decel_radps2"
        else:
            continue
        values = distributions[name]
        low = float(values["p10"] if training else values["p50"])
        high = float(values["p90"] if training else values["p50"])
        setattr(drive, f"{prefix}_min", low)
        setattr(drive, f"{prefix}_max", high)
    quantized = calibration["delay_quantization"]
    drive.min_delay_policy_steps = int(
        quantized["p10_steps"] if training else quantized["p50_steps"]
    )
    drive.max_delay_policy_steps = int(
        quantized["p90_steps"] if training else quantized["p50_steps"]
    )
    drive.randomize_dynamics = bool(training)
    drive.randomize_delay = bool(training)
    if training:
        mass_range = tuple(mass * factor for factor in MASS_RELATIVE_RANGE)
        static_range = tuple(
            static_friction * factor for factor in FRICTION_RELATIVE_RANGE
        )
        dynamic_range = tuple(
            static_friction * DYNAMIC_FRICTION_RATIO * factor
            for factor in FRICTION_RELATIVE_RANGE
        )
    else:
        mass_range = (mass, mass)
        static_range = (static_friction, static_friction)
        dynamic = static_friction * DYNAMIC_FRICTION_RATIO
        dynamic_range = (dynamic, dynamic)
    env_cfg.events.total_mass.params["total_mass_range_kg"] = mass_range
    wheel_material = env_cfg.events.wheel_material.params
    wheel_material["static_friction_range"] = static_range
    wheel_material["dynamic_friction_range"] = dynamic_range
    wheel_material["make_consistent"] = True
    return {
        "source": "real_car",
        "run_id": calibration.get("run_id"),
        "training_randomization": bool(training),
        "total_mass_range_kg": list(mass_range),
        "wheel_static_friction_range": list(static_range),
        "wheel_dynamic_friction_range": list(dynamic_range),
        "dynamic_friction_ratio": DYNAMIC_FRICTION_RATIO,
        "caster_friction": CASTER_FRICTION,
        "mass_distribution": {
            "base_link": 0.825,
            "left_wheel_link": 0.050,
            "right_wheel_link": 0.050,
            "imu_link": 0.00625,
            "laser_stand_link": 0.0125,
            "laser_link": 0.03125,
            "front_caster_link": 0.0125,
            "back_caster_link": 0.0125,
        },
    }


def production_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    """将 resolved 数据还原为 bundle schema；人工字段为空时门禁会明确拒绝。"""

    return {
        "schema_version": 1,
        "measured_at": calibration.get("measured_at"),
        **dict(calibration["calibration"]),
        **dict(calibration.get("manual_fields", {})),
        "source_log_sha256": calibration["files"]["calibration_csv_sha256"],
    }


def verify_production_overlay(
    overlay: dict[str, Any], calibration: dict[str, Any] | None
) -> dict[str, Any]:
    """旧 checkpoint 可补录人工字段；已用于训练的物理字段禁止替换。"""

    if calibration is None:
        return overlay
    expected = production_calibration(calibration)
    for name in (*METRICS, "source_log_sha256"):
        if overlay.get(name) != expected.get(name):
            raise CalibrationError(f"生产标定文件的 {name} 与 checkpoint 不一致")
    for name in ("total_mass_kg", "wheel_friction_center"):
        if expected.get(name) is not None and overlay.get(name) != expected.get(name):
            raise CalibrationError(f"生产标定文件的 {name} 与 checkpoint 不一致")
    return overlay
