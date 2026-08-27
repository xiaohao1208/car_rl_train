"""记录训练与导出环境的可复现性锁文件。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate_bundle import sha256_file


def _package_version(name: str) -> str | None:
    """读取已安装 Python 包版本，未安装时返回空值。"""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(path: Path) -> str | None:
    """读取指定仓库当前提交；路径不是 Git 仓库时返回空值。"""

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def capture_environment_lock(
    project_root: str | Path,
    isaaclab_root: str | Path | None,
    source_fingerprint_sha256: str,
    configuration_files: list[str | Path],
) -> dict[str, Any]:
    """采集系统、GPU、关键依赖、源码和配置摘要，形成环境锁。"""

    project = Path(project_root).resolve()
    isaaclab = Path(isaaclab_root).resolve() if isaaclab_root is not None else None
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda_version = torch.version.cuda
    except ImportError:
        gpu_name, cuda_version = None, None
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "gpu_name": gpu_name,
        "nvidia_driver": driver.stdout.strip().splitlines()[0] if driver.returncode == 0 else None,
        "cuda_version": cuda_version,
        "packages": {
            name: _package_version(name)
            for name in (
                "isaacsim",
                "isaaclab",
                "rsl-rl-lib",
                "torch",
                "onnx",
                "onnxruntime",
                "gymnasium",
            )
        },
        "isaaclab_commit": _git_commit(isaaclab) if isaaclab is not None else None,
        "training_project_commit": _git_commit(project),
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "configuration_sha256": {
            Path(path).resolve().relative_to(project).as_posix(): sha256_file(path)
            for path in configuration_files
        },
    }


def write_environment_lock(path: str | Path, value: dict[str, Any]) -> None:
    """以稳定排序的 JSON 格式写入环境锁。"""

    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
