"""命令行、PyCharm、训练、评估和导出共享的工作流配置。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN_STATUS_NAME = "run_status.json"
CHECKPOINT_METADATA_SCHEMA_VERSION = 2


def next_training_segment_iterations(
    cumulative_iterations: int,
    segments_completed: int,
    initial_iterations: int,
    extension_iterations: int,
    max_total_iterations: int,
) -> int:
    """返回下一训练段预算；到达累计上限时返回零。"""

    values = (
        cumulative_iterations,
        segments_completed,
        initial_iterations,
        extension_iterations,
        max_total_iterations,
    )
    if any(int(value) < 0 for value in values[:2]) or any(
        int(value) <= 0 for value in values[2:]
    ):
        raise ValueError("training segment budgets are invalid")
    if max_total_iterations < initial_iterations:
        raise ValueError("maximum training budget is smaller than initial budget")
    remaining = max_total_iterations - cumulative_iterations
    if remaining <= 0:
        return 0
    requested = initial_iterations if segments_completed == 0 else extension_iterations
    return min(requested, remaining)


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> Path:
    """在目标目录内原子发布 JSON，避免中断留下半写文件。"""

    destination = Path(path)
    pending = destination.with_name(
        f".{destination.name}.pending.{os.getpid()}"
    )
    pending.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, destination)
    return destination


def load_run_status(run_dir: str | Path) -> dict[str, Any] | None:
    """读取运行状态；旧目录或损坏状态返回 None。"""

    path = Path(run_dir) / RUN_STATUS_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    return value


def completed_run_checkpoint(run_dir: str | Path) -> Path | None:
    """仅解析由新训练入口完整发布的最终 checkpoint。"""

    directory = Path(run_dir).resolve()
    status = load_run_status(directory)
    if status is None or status.get("status") != "completed":
        return None
    name = status.get("final_checkpoint")
    if not isinstance(name, str) or Path(name).name != name:
        return None
    checkpoint = (directory / name).resolve()
    if checkpoint.parent != directory or not checkpoint.is_file():
        return None
    return checkpoint


def eligible_run_checkpoints(run_dir: str | Path) -> list[Path]:
    """返回训练资格门禁发布的最终阶段候选；缺失或越界时拒绝。"""

    directory = Path(run_dir).resolve()
    status = load_run_status(directory)
    if (
        status is None
        or status.get("status") != "completed"
        or status.get("training_eligible") is not True
    ):
        return []
    names = status.get("eligible_checkpoints")
    if not isinstance(names, list):
        return []
    result = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            return []
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            return []
        result.append(path)
    return result


def load_workflow_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML，并将内部相对路径统一解释为项目根目录相对路径。"""

    config_path = resolve_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    value["_config_path"] = project_relative_path(config_path)
    return value


def resolve_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """解析绝对或项目相对路径。"""

    path = Path(value).expanduser()
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def project_relative_path(value: str | Path) -> str:
    """把项目内路径编码为可移植的 POSIX 相对路径。"""

    path = resolve_path(value)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"路径必须位于项目目录内: {value}") from error


def portable_project_value(value: Any) -> Any:
    """递归移除结构化元数据中的项目绝对路径。"""

    if isinstance(value, dict):
        return {key: portable_project_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_project_value(item) for item in value]
    if isinstance(value, tuple):
        return [portable_project_value(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                anchors = {
                    "assets",
                    "calibration",
                    "configs",
                    "evaluations",
                    "exports",
                    "logs",
                    "observations",
                    "scripts",
                    "source",
                }
                for index, part in enumerate(candidate.parts):
                    if part in anchors:
                        return Path(*candidate.parts[index:]).as_posix()
                return candidate.name
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并映射，供 CLI 覆盖 YAML 使用。"""

    result = deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_object_config(target: Any, values: dict[str, Any], label: str) -> None:
    """把 YAML 映射写入 Isaac configclass，并拒绝拼错的字段。"""

    for key, value in values.items():
        if not hasattr(target, key):
            raise ValueError(f"unknown {label} option: {key}")
        setattr(target, key, value)


def resolve_checkpoint(value: str | Path, experiment_name: str = "car_local_navigation") -> Path:
    """解析明确 checkpoint，或选择最近完整运行的最终 checkpoint。"""

    if str(value).lower() != "latest":
        path = resolve_path(value, must_exist=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = ROOT / "logs" / "rsl_rl" / experiment_name
    candidates = [
        checkpoint
        for run_dir in root.iterdir()
        if run_dir.is_dir()
        if eligible_run_checkpoints(run_dir)
        for checkpoint in [completed_run_checkpoint(run_dir)]
        if checkpoint is not None
    ] if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no completed run checkpoint found under {root}; "
            "use an explicit checkpoint for legacy runs"
        )
    return max(candidates, key=lambda path: path.parent.stat().st_mtime)


def resolve_checkpoint_selection(value: str | Path) -> tuple[Path, Path | None]:
    """解析 checkpoint、评估汇总或 latest_evaluation 指针。"""

    if str(value).lower() == "latest":
        return resolve_checkpoint("latest"), None
    path = resolve_path(value, must_exist=True)
    if path.suffix.lower() != ".json":
        return resolve_checkpoint(path), None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 选择文件根节点必须是 mapping: {path}")
    if payload.get("schema_version") == 1 and payload.get("evaluation"):
        evaluation = resolve_path(payload["evaluation"], must_exist=True)
        nested = json.loads(evaluation.read_text(encoding="utf-8"))
        checkpoint = nested.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"评估没有通过生产门禁或未选择 checkpoint: {evaluation}")
        return resolve_checkpoint(checkpoint), evaluation
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"选择文件缺少 checkpoint: {path}")
    return resolve_checkpoint(checkpoint), path


def checkpoint_metadata(payload: dict[str, Any], checkpoint: str | Path) -> dict[str, Any] | None:
    """只接受当前生产训练入口写入的内嵌元数据。"""

    del checkpoint
    infos = payload.get("infos")
    metadata = infos.get("car_rl") if isinstance(infos, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION
        or metadata.get("policy", {}).get("name") != "split_fusion"
        or metadata.get("algorithm", {}).get("name") != "ppo"
    ):
        return None
    return metadata
