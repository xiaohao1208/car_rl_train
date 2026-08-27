"""[必选] 检查训练环境、配置、资产、标定和可选模型产物。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml

from car_rl_train.assets import load_and_validate_asset_manifest, resolve_project_path, sha256_file
from car_rl_train.workflow import ROOT, project_relative_path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_onnx(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("ONNX 必须只有一个输入和一个输出")

    def tensor(value) -> dict:
        return {
            "name": value.name,
            "shape": [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim],
        }

    result = {
        "path": project_relative_path(path),
        "opset": max(item.version for item in model.opset_import),
        "input": tensor(model.graph.input[0]),
        "output": tensor(model.graph.output[0]),
        "sha256": sha256_file(path),
    }
    expected = {
        "opset": 18,
        "input": {"name": "observation", "shape": [1, 86]},
        "output": {"name": "action", "shape": [1, 2]},
    }
    for key, value in expected.items():
        if result[key] != value:
            raise RuntimeError(f"ONNX 契约错误 {key}: {result[key]!r}")
    return result


def validate_configs() -> dict[str, int]:
    """验证项目所有 YAML 的根节点和 schema_version。"""

    schemas: dict[str, int] = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"配置根节点必须是 mapping: {path}")
        schema = value.get("schema_version")
        if not isinstance(schema, int) or schema < 1:
            raise ValueError(f"配置缺少有效 schema_version: {path}")
        schemas[path.relative_to(ROOT).as_posix()] = schema
    return schemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise RuntimeError(f"训练配置必须使用 schema_version=2: {config_path}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"需要 Python 3.11，当前为 {platform.python_version()}")

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    versions = {
        name: package_version(name)
        for name in (
            "isaacsim", "isaaclab", "isaaclab-rl", "isaaclab-tasks",
            "rsl-rl-lib", "torch", "gymnasium", "numpy", "h5py",
            "tensordict", "PyYAML", "psutil", "onnx", "onnxruntime",
        )
    }
    expected_versions = {
        "rsl-rl-lib": "3.1.2",
        "numpy": "1.26.0",
        "h5py": "3.16.0",
        "tensordict": "0.13.0",
    }
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in expected_versions.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"依赖版本不匹配: {mismatches}")
    if not str(versions.get("isaacsim", "")).startswith("5.1"):
        raise RuntimeError(f"需要 Isaac Sim 5.1，当前为 {versions.get('isaacsim')}")

    from car_rl_train.calibration import load_calibration

    calibration = load_calibration(config.get("calibration"))
    manifest = load_and_validate_asset_manifest()
    checkpoint_report = None
    if args.checkpoint is not None:
        checkpoint = resolve_project_path(args.checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("checkpoint 根节点必须是 mapping")
        checkpoint_report = {
            "path": project_relative_path(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    onnx_report = check_onnx(resolve_project_path(args.onnx)) if args.onnx else None
    report = {
        "status": "ok",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": versions,
        "configs": validate_configs(),
        "training_config": project_relative_path(config_path),
        "calibration": calibration.get("run_id") if calibration else None,
        "asset": manifest,
        "checkpoint": checkpoint_report,
        "onnx": onnx_report,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
