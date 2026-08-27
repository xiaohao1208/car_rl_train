"""生成带生产门禁的控制器专用 schema v2 模型包。"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml

from ..contract.schema import (
    BUNDLE_SCHEMA_VERSION,
    INPUT_NAME,
    INPUT_SHAPE,
    ONNX_OPSET,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
)
from .validate_bundle import sha256_file, validate_bundle

CALIBRATION_FIELDS = (
    "measured_at",
    "total_mass_kg",
    "motor_time_constant_sec",
    "action_delay_sec",
    "left_motor_gain",
    "right_motor_gain",
    "linear_accel_mps2",
    "linear_decel_mps2",
    "angular_accel_radps2",
    "angular_decel_radps2",
    "wheel_friction_center",
    "source_log_sha256",
)

PRODUCTION_GATES = {
    "success_rate": (">=", 0.95),
    "collision_rate": ("<=", 0.01),
    "stall_rate": ("<=", 0.02),
    "p95_cross_track_error_m": ("<=", 0.10),
    "mean_goal_position_error_m": ("<=", 0.08),
    "mean_goal_yaw_error_rad": ("<=", 0.14),
    "pytorch_onnx_max_abs_error": ("<=", 1.0e-5),
    "onnxruntime_p99_ms": ("<=", 10.0),
}


def validate_production_inputs(calibration: dict[str, Any], evaluation: dict[str, Any]) -> None:
    """验证真实校准与正式评估是否满足生产打包条件。"""

    missing = [field for field in CALIBRATION_FIELDS if calibration.get(field) is None]
    if missing:
        raise ValueError(f"production calibration is incomplete: {', '.join(missing)}")
    failures: list[str] = []
    suites = evaluation.get("suite_metrics")
    if not isinstance(suites, dict):
        failures.append("layered core/dynamic suite_metrics are missing")
        core = evaluation
        dynamic = {}
    else:
        core = suites.get("core") or {}
        dynamic = suites.get("dynamic") or {}
    for name, (operator, threshold) in PRODUCTION_GATES.items():
        source = evaluation if name.startswith(("pytorch_", "onnxruntime_")) else core
        raw_value = source.get(name)
        if raw_value is None:
            failures.append(f"{name}=missing")
            continue
        value = float(raw_value)
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value} requires {operator}{threshold}")
    if int(core.get("episodes", 0)) < 1000:
        failures.append("core episodes must be at least 1000")
    if float(dynamic.get("success_rate", -1.0)) < 0.85:
        failures.append("dynamic success_rate must be at least 0.85")
    if float(dynamic.get("collision_rate", 1.0)) > 0.03:
        failures.append("dynamic collision_rate must not exceed 0.03")
    if float(dynamic.get("stall_rate", 1.0)) > 0.05:
        failures.append("dynamic stall_rate must not exceed 0.05")
    if int(dynamic.get("episodes", 0)) < 500:
        failures.append("dynamic episodes must be at least 500")
    if int(evaluation.get("real_test_observation_count", 0)) < 4000:
        failures.append("real_test_observation_count must be at least 4000")
    source_digest = evaluation.get("real_observation_source_sha256")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        failures.append("real_observation_source_sha256 must be a 64-character SHA-256")
    if core.get("test_seed_range") != [20000, 20999]:
        failures.append("core test_seed_range must equal [20000, 20999]")
    if dynamic.get("test_seed_range") != [22000, 22499]:
        failures.append("dynamic test_seed_range must equal [22000, 22499]")
    if failures:
        raise ValueError("production evaluation gates failed: " + "; ".join(failures))


def create_bundle(
    model_path: str | Path,
    evaluation: dict[str, Any],
    environment_lock: dict[str, Any],
    calibration: dict[str, Any],
    output_dir: str | Path,
    model_version: str,
) -> Path:
    """创建不可覆盖的 schema v2 目录包、清单、摘要和 ZIP 文件。"""

    validate_production_inputs(calibration, evaluation)
    if not model_version or any(character in model_version for character in "\\/:*?\"<>|"):
        raise ValueError("model_version is empty or contains unsafe path characters")
    source_model = Path(model_path)
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {root}")
    controller = root / "controller"
    controller.mkdir(parents=True)
    destination_model = controller / "car_rl_model.onnx"
    shutil.copy2(source_model, destination_model)
    metadata = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "role": "controller",
        "model_version": model_version,
        "model_file": "car_rl_model.onnx",
        "onnx_opset": ONNX_OPSET,
        "onnx_sha256": sha256_file(destination_model),
        "max_inference_ms": 10.0,
        "input": {"name": INPUT_NAME, "shape": INPUT_SHAPE},
        "output": {"name": OUTPUT_NAME, "shape": OUTPUT_SHAPE},
    }
    (root / "bundle.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_version": model_version,
                "controller_metadata": "controller/metadata.yaml",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (controller / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (controller / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "environment-lock.json").write_text(
        json.dumps(environment_lock, indent=2, sort_keys=True), encoding="utf-8"
    )
    validate_bundle(root)
    return root


def archive_bundle(bundle: str | Path, archive_path: str | Path | None = None) -> tuple[Path, Path]:
    """把已校验目录包压成根级 ZIP，并生成相邻 SHA-256 文件。"""

    root = Path(bundle).resolve()
    validate_bundle(root)
    archive = Path(archive_path) if archive_path else root.parent / f"car_rl_model_bundle_{root.name.removeprefix('car_rl_bundle_v2_')}.zip"
    archive = archive.resolve()
    checksum = Path(str(archive) + ".sha256")
    if archive.exists() or checksum.exists():
        raise FileExistsError(f"refusing to overwrite archive or checksum: {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stream.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, checksum
