"""小车描述资产的路径、摘要和清单校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .workflow import ROOT


ASSET_CONFIG_PATH = ROOT / "configs" / "asset.yaml"
ASSET_MANIFEST_PATH = ROOT / "assets" / "asset_manifest.json"


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asset_config(path: str | Path = ASSET_CONFIG_PATH) -> dict[str, Any]:
    """读取并验证资产转换配置。"""

    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"资产配置必须使用 schema_version=1: {config_path}")
    required = ("source_xacro", "expanded_urdf", "output_usd", "package_name")
    missing = [name for name in required if not value.get(name)]
    if missing:
        raise ValueError(f"资产配置缺少字段: {missing}")
    return value


def resolve_project_path(value: str | Path) -> Path:
    """将项目相对路径解析为受项目根目录约束的绝对路径。"""

    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else ROOT / raw).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"路径超出项目目录: {value}")
    return path


def source_files(config: dict[str, Any]) -> list[Path]:
    """返回参与 Xacro 展开的所有本地源文件。"""

    source = resolve_project_path(config["source_xacro"])
    root = source.parent
    expanded = resolve_project_path(config["expanded_urdf"])
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != expanded and path.suffix in {".xacro", ".urdf", ".stl", ".dae", ".obj"}
    )
    if source not in files:
        raise FileNotFoundError(source)
    return files


def source_fingerprint(files: list[Path], root: Path) -> str:
    """根据相对路径和内容生成稳定的描述源指纹。"""

    digest = hashlib.sha256()
    for path in files:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_asset_manifest(
    config: dict[str, Any],
    *,
    converter: dict[str, Any],
) -> dict[str, Any]:
    """根据已经发布的 URDF/USD 构造 schema v2 清单。"""

    xacro_path = resolve_project_path(config["source_xacro"])
    source_root = xacro_path.parent
    urdf_path = resolve_project_path(config["expanded_urdf"])
    usd_path = resolve_project_path(config["output_usd"])
    files = source_files(config)
    for path in (urdf_path, usd_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return {
        "schema_version": 2,
        "source_fingerprint_sha256": source_fingerprint(files, source_root),
        "sources": {
            path.relative_to(source_root).as_posix(): sha256_file(path)
            for path in files
        },
        "urdf": {
            "path": urdf_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(urdf_path),
        },
        "usd": {
            "path": usd_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(usd_path),
        },
        "asset_config_sha256": sha256_file(ASSET_CONFIG_PATH),
        "converter": converter,
    }


def load_and_validate_asset_manifest() -> dict[str, Any]:
    """验证清单、源描述、展开 URDF、最终 USD 和转换配置完全一致。"""

    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("assets/asset_manifest.json 必须使用 schema_version=2")
    config = load_asset_config()
    files = source_files(config)
    source_root = resolve_project_path(config["source_xacro"]).parent
    expected_sources = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in files
    }
    checks = {
        "sources": expected_sources,
        "source_fingerprint_sha256": source_fingerprint(files, source_root),
        "asset_config_sha256": sha256_file(ASSET_CONFIG_PATH),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"资产清单字段与当前文件不一致: {key}")
    for name, config_key in (("urdf", "expanded_urdf"), ("usd", "output_usd")):
        path = resolve_project_path(config[config_key])
        entry = manifest.get(name)
        expected = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        if entry != expected:
            raise ValueError(f"资产清单字段与当前文件不一致: {name}")
    return manifest
