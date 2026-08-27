"""Jetson 控制器专用模型包 schema 的严格读取与校验。"""

from __future__ import annotations

import hashlib
import json
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
from .export_onnx import inspect_onnx_contract


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免一次性加载大型模型文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    """解析清单相对路径，并阻止路径穿越模型包根目录。"""

    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError(f"path escapes bundle root: {relative}")
    return candidate


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 映射，拒绝其他根节点类型。"""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return value


def validate_bundle(bundle: str | Path, check_onnx: bool = True) -> dict[str, Any]:
    """校验模型包 schema、文件摘要、ONNX 契约和控制元数据。"""

    root = Path(bundle).resolve()
    manifest = _read_yaml(root / "bundle.yaml")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("only bundle schema_version 2 is supported")
    bundle_version = manifest.get("bundle_version")
    if not isinstance(bundle_version, str) or not bundle_version:
        raise ValueError("bundle_version must be non-empty")
    metadata_path = _safe_child(root, manifest.get("controller_metadata", ""))
    metadata = _read_yaml(metadata_path)
    expected = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "role": "controller",
        "model_version": bundle_version,
        "onnx_opset": ONNX_OPSET,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"invalid metadata {key}")
    if metadata.get("input") != {"name": INPUT_NAME, "shape": INPUT_SHAPE}:
        raise ValueError("invalid controller input")
    if metadata.get("output") != {"name": OUTPUT_NAME, "shape": OUTPUT_SHAPE}:
        raise ValueError("invalid controller output")
    model_path = _safe_child(metadata_path.parent, metadata.get("model_file", ""))
    digest = sha256_file(model_path)
    if metadata.get("onnx_sha256") != digest:
        raise ValueError("ONNX SHA-256 mismatch")
    evaluation_path = root / "controller" / "evaluation.json"
    lock_path = root / "environment-lock.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    environment_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not evaluation or not environment_lock:
        raise ValueError("evaluation and environment lock must be non-empty")
    if check_onnx:
        inspect_onnx_contract(model_path)
    return {
        "bundle_version": bundle_version,
        "model_path": model_path.relative_to(root).as_posix(),
        "onnx_sha256": digest,
        "evaluation": evaluation,
        "environment_lock": environment_lock,
    }
