"""[必选] 从项目内 Xacro 描述生成经过校验的 Isaac Sim USD 资产。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from car_rl_train.assets import (
    ASSET_MANIFEST_PATH,
    build_asset_manifest,
    load_asset_config,
    resolve_project_path,
)
from car_rl_train.workflow import ROOT

os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT / "configs" / "asset.yaml")
parser.add_argument("--xacro", type=Path, default=Path("/opt/ros/humble/bin/xacro"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cpu")
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import Sdf, Usd  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


FORBIDDEN_ASSET_MARKERS = (
    b"car_rl_" + b"training",
    b"ros2_car_gpt",
    b"/home/",
    b"/" + b"Users/",
    b"c:" + b"\\Users\\",
)


def _reject_private_markers(path: Path) -> None:
    """拒绝把本机目录或旧项目名写入公开资产。"""

    payload = path.read_bytes()
    matches = [
        marker.decode("utf-8", errors="replace")
        for marker in FORBIDDEN_ASSET_MARKERS
        if marker in payload
    ]
    if matches:
        raise RuntimeError(f"资产包含不可发布的路径标记: {matches}")


def _xacro_environment(prefix: Path) -> dict[str, str]:
    """构造只把本地描述目录注册为 car_description 的 Xacro 环境。"""

    environment = dict(os.environ)
    ros_root = args.xacro.resolve().parents[1]
    python_paths = [
        ros_root / "lib" / "python3.10" / "site-packages",
        ros_root / "local" / "lib" / "python3.10" / "dist-packages",
    ]
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in python_paths if path.is_dir()]
        + ([existing] if existing else [])
    )
    environment["AMENT_PREFIX_PATH"] = os.pathsep.join(
        [str(prefix), str(ros_root)]
    )
    return environment


def _expand_xacro(config: dict, staging: Path) -> Path:
    """在隔离的临时 ament 索引中展开本地 Xacro。"""

    source = resolve_project_path(config["source_xacro"])
    if not args.xacro.is_file():
        raise FileNotFoundError(f"找不到 Xacro 可执行文件: {args.xacro}")
    package_name = str(config["package_name"])
    prefix = staging / "ament"
    resource = prefix / "share" / "ament_index" / "resource_index" / "packages"
    package_share = prefix / "share" / package_name
    resource.mkdir(parents=True)
    package_share.mkdir(parents=True)
    (resource / package_name).touch()
    (package_share / "urdf").symlink_to(source.parent, target_is_directory=True)
    expanded = staging / "car.urdf"
    result = subprocess.run(
        [str(args.xacro.resolve()), str(source), "-o", str(expanded)],
        capture_output=True,
        text=True,
        check=False,
        env=_xacro_environment(prefix),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Xacro 展开失败: {detail}")
    text = expanded.read_text(encoding="utf-8")
    robot_offset = text.find("<robot")
    declaration_end = text.find("\n")
    if robot_offset < 0 or declaration_end < 0:
        raise RuntimeError("Xacro 输出缺少 XML 声明或 robot 根节点")
    text = text[: declaration_end + 1] + text[robot_offset:]
    expanded.write_text(text, encoding="utf-8")
    if "$(find " in text or "package://" in text:
        raise RuntimeError("展开后的 URDF 仍包含未解析的包路径")
    if "<robot" not in text or "left_wheel_joint" not in text or "right_wheel_joint" not in text:
        raise RuntimeError("展开后的 URDF 缺少机器人或驱动轮关节")
    _reject_private_markers(expanded)
    return expanded


def _convert_usd(config: dict, urdf: Path, staging: Path) -> Path:
    """转换并扁平化 USD，同时验证关节、默认 prim 和外部依赖。"""

    settings = config.get("converter") or {}
    converter_config = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(staging / "converter"),
        usd_file_name="car.usd",
        fix_base=bool(settings.get("fix_base", False)),
        merge_fixed_joints=bool(settings.get("merge_fixed_joints", False)),
        force_usd_conversion=bool(settings.get("force_usd_conversion", True)),
        self_collision=bool(settings.get("self_collision", False)),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type=str(settings.get("drive_type", "force")),
            target_type={
                "left_wheel_joint": "velocity",
                "right_wheel_joint": "velocity",
                ".*": "none",
            },
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    "left_wheel_joint": float(settings.get("stiffness", 20.0)),
                    "right_wheel_joint": float(settings.get("stiffness", 20.0)),
                    ".*": 0.0,
                },
                damping=float(settings.get("damping", 0.0)),
            ),
        ),
    )
    generated = staging / "converter" / "car.usd"
    try:
        converter = UrdfConverter(converter_config)
    except SystemExit as error:
        if error.code not in (None, 0) or not generated.is_file():
            raise RuntimeError("Isaac Sim URDF 转换器异常退出") from error
    else:
        generated = Path(converter.usd_path)
    stage = Usd.Stage.Open(str(generated))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError("转换后的 USD 缺少有效 defaultPrim")
    prim_names = {prim.GetName() for prim in stage.Traverse()}
    missing = set(config.get("required_prims") or ()) - prim_names
    if missing:
        raise RuntimeError(f"转换后的 USD 缺少必要 prim: {sorted(missing)}")
    flattened = staging / "car_flattened.usd"
    flattened_layer = stage.Flatten()
    layer_text = flattened_layer.ExportToString()
    replacements = {
        str(staging): ".",
        str(ROOT): ".",
        (b"car_rl_" + b"training").decode(): "car_rl_train",
    }
    for source_text, portable_text in replacements.items():
        layer_text = layer_text.replace(source_text, portable_text)
    sanitized_layer = Sdf.Layer.CreateAnonymous("car_rl_train.usda")
    if not sanitized_layer.ImportFromString(layer_text):
        raise RuntimeError("无法重建已清理路径信息的 USD 图层")
    sanitized_layer.comment = "car_rl_train flattened robot asset"
    if not sanitized_layer.Export(str(flattened)):
        raise RuntimeError("USD 扁平化导出失败")
    sanitized_layer = None
    flattened_layer = None
    stage = None
    gc.collect()
    layer = Sdf.Layer.FindOrOpen(str(flattened))
    if layer is None:
        raise RuntimeError("无法重新打开扁平化 USD")
    dependencies = sorted(
        path_text
        for path in layer.GetExternalReferences()
        if (path_text := str(path))
    )
    if dependencies:
        raise RuntimeError(f"最终 USD 仍包含外部依赖: {dependencies}")
    layer = None
    gc.collect()
    _reject_private_markers(flattened)
    return flattened


def _publish(config: dict, urdf: Path, usd: Path, staging: Path) -> dict:
    """先完整生成清单，再以可回滚事务发布三个目标文件。"""

    urdf_target = resolve_project_path(config["expanded_urdf"])
    usd_target = resolve_project_path(config["output_usd"])
    urdf_pending = staging / "published.urdf"
    usd_pending = staging / "published.usd"
    manifest_pending = staging / "asset_manifest.json"
    shutil.copy2(urdf, urdf_pending)
    shutil.copy2(usd, usd_pending)

    # 清单必须按最终项目路径计算，因此先临时放置目标文件并在异常时恢复。
    targets = (urdf_target, usd_target, ASSET_MANIFEST_PATH)
    pending = (urdf_pending, usd_pending, manifest_pending)
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for target in targets[:2]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = staging / f"backup-{target.name}"
                os.replace(target, backup)
                backups.append((backup, target))
        os.replace(urdf_pending, urdf_target)
        installed.append(urdf_target)
        os.replace(usd_pending, usd_target)
        installed.append(usd_target)
        manifest = build_asset_manifest(
            config,
            converter={
                "name": "Isaac Lab UrdfConverter",
                "settings": config.get("converter") or {},
            },
        )
        manifest_pending.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if ASSET_MANIFEST_PATH.exists():
            backup = staging / "backup-asset_manifest.json"
            os.replace(ASSET_MANIFEST_PATH, backup)
            backups.append((backup, ASSET_MANIFEST_PATH))
        os.replace(manifest_pending, ASSET_MANIFEST_PATH)
        installed.append(ASSET_MANIFEST_PATH)
        return manifest
    except Exception:
        for target in reversed(installed):
            target.unlink(missing_ok=True)
        for backup, target in reversed(backups):
            os.replace(backup, target)
        raise


def main() -> None:
    """执行展开、转换、验证和事务发布。"""

    config = load_asset_config(args.config)
    cache_root = ROOT / "assets" / ".asset_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=cache_root) as directory:
        staging = Path(directory)
        urdf = _expand_xacro(config, staging)
        usd = _convert_usd(config, urdf, staging)
        manifest = _publish(config, urdf, usd, staging)
    try:
        cache_root.rmdir()
    except OSError:
        pass
    print(f"urdf={resolve_project_path(config['expanded_urdf'])}")
    print(f"usd={resolve_project_path(config['output_usd'])}")
    print(f"source_sha256={manifest['source_fingerprint_sha256']}")
    print(f"usd_sha256={manifest['usd']['sha256']}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - Isaac Sim 关闭流程会吞掉原始异常
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)
    else:
        simulation_app.close()
