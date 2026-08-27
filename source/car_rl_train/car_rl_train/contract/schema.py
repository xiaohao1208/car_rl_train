"""控制器专用 schema v2 的常量与校验器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BUNDLE_SCHEMA_VERSION = 2
OBSERVATION_CONTRACT_VERSION = 1
OBSERVATION_SIZE = 86
ACTION_SIZE = 2
ONNX_OPSET = 18
INPUT_NAME = "observation"
OUTPUT_NAME = "action"
INPUT_SHAPE = [1, OBSERVATION_SIZE]
OUTPUT_SHAPE = [1, ACTION_SIZE]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 映射，并拒绝列表、标量等不符合契约的根节点。"""

    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def validate_handoff_manifest(manifest: dict[str, Any]) -> None:
    """校验交接清单中的 schema 版本及观测、动作维度。"""

    expected = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"handoff {key}: expected {value}, got {manifest.get(key)}")
