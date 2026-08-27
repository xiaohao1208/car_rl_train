"""[必选] 根据正式评估结果导出并发布固定名称的生产模型。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from datetime import date
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import torch
import yaml

from car_rl_train.calibration import (
    production_calibration,
    select_calibration,
    verify_production_overlay,
)
from car_rl_train.contract.schema import OBSERVATION_CONTRACT_VERSION
from car_rl_train.export.environment import capture_environment_lock, write_environment_lock
from car_rl_train.export.export_onnx import export_controller, validate_onnx_numerics
from car_rl_train.export.make_bundle import archive_bundle, create_bundle
from car_rl_train.export.validate_bundle import sha256_file, validate_bundle
from car_rl_train.models import CarActorCritic
from car_rl_train.workflow import (
    ROOT,
    checkpoint_metadata,
    load_workflow_config,
    portable_project_value,
    resolve_checkpoint,
    resolve_checkpoint_selection,
    resolve_path,
)


def read_mapping(path: Path) -> dict:
    """读取 JSON/YAML 映射配置。"""

    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def discover_isaaclab_root(configured: str | Path | None) -> Path | None:
    """从显式配置、环境变量或已安装模块解析 Isaac Lab 根目录。"""

    candidate = configured or os.environ.get("ISAACLAB_ROOT")
    if candidate:
        path = Path(candidate).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin).resolve()
    for parent in (origin.parent, *origin.parents):
        if (parent / "isaaclab.sh").is_file() or (parent / "isaaclab.bat").is_file():
            return parent
    return None


def resolve_selection(value: str | Path) -> tuple[Path, Path | None]:
    """解析 checkpoint 或评估产生的 best_checkpoint.json。"""

    if str(value).lower() == "latest":
        pointer = ROOT / "evaluations" / "latest_evaluation.json"
        return resolve_checkpoint_selection(pointer if pointer.is_file() else "latest")
    path = resolve_path(value, must_exist=True)
    if path.suffix.lower() == ".json":
        selection = read_mapping(path)
        if selection.get("schema_version") == 1 and selection.get("evaluation"):
            return resolve_checkpoint_selection(path)
        checkpoint = resolve_checkpoint(selection["checkpoint"])
        if selection.get("schema_version") == 2 and "suite_metrics" in selection:
            return checkpoint, path
        return checkpoint, resolve_path(selection["evaluation"], must_exist=True)
    return resolve_checkpoint(path), None


def load_actor(checkpoint_path: Path):
    """从当前生产 checkpoint 恢复并提取确定性 actor。"""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint_metadata(payload, checkpoint_path)
    if metadata is None:
        raise ValueError("checkpoint不是当前split_fusion生产格式")
    kwargs = dict(metadata.get("policy", {}).get("kwargs", {}))
    policy = CarActorCritic(**kwargs)
    state = payload.get("model_state_dict", payload)
    policy.load_state_dict(state)
    return policy.export_actor().eval(), metadata, "split_fusion"


def validate_real_observation_manifest(observation_path: Path, manifest_path: Path) -> dict:
    """验证真实车辆观测的契约、数量、来源声明和文件摘要。"""

    manifest = read_mapping(manifest_path)
    expected = {
        "schema_version": 1,
        "source": "real_car",
        "production_eligible": True,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"invalid real observation manifest {key}: {manifest.get(key)!r}")
    digest = sha256_file(observation_path)
    if manifest.get("sha256") != digest:
        raise ValueError("real observation file SHA-256 does not match manifest")
    if int(manifest.get("count", 0)) < 4000:
        raise ValueError("real observation manifest count must be at least 4000")
    return manifest


def validation_observations(path: Path | None, manifest_path: Path | None) -> tuple[np.ndarray, int, str | None]:
    """组合真实、边界和随机观测，构造恰好 10,000 条验证输入。"""

    if path:
        if manifest_path is None:
            raise ValueError("real observations require observation_manifest")
        manifest = validate_real_observation_manifest(path, manifest_path)
        payload = np.load(path)
        values = payload["observation"] if "observation" in payload else payload[payload.files[0]]
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 86 or values.shape[0] < 4000:
            raise ValueError("real observations must have shape [N,86] with N >= 4000")
        if int(manifest["count"]) != int(values.shape[0]):
            raise ValueError("real observation manifest count does not match NPZ")
        values = values[:4000]
        real_count = 4000
        source_digest = str(manifest["sha256"])
    else:
        values = np.empty((0, 86), dtype=np.float32)
        real_count = 0
        source_digest = None
    rng = np.random.default_rng(20260730)
    boundary = rng.choice(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), size=(3000, 86))
    random_values = rng.uniform(-1.0, 1.0, size=(3000, 86)).astype(np.float32)
    if real_count:
        return np.concatenate((values, boundary, random_values)), real_count, source_digest
    extra = rng.uniform(-1.0, 1.0, size=(4000, 86)).astype(np.float32)
    return np.concatenate((extra, boundary, random_values)), 0, None


def validate_release_directory(bundle: Path) -> None:
    """校验目录包、数值报告、ZIP 摘要以及压缩前后的逐文件一致性。"""

    result = validate_bundle(bundle)
    numerical_path = bundle / "onnx_validation.json"
    archive_path = bundle / "car_rl_model.zip"
    checksum_path = bundle / "car_rl_model.zip.sha256"
    for path in (numerical_path, archive_path, checksum_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"生产模型缺少文件: {path}")
    numerical = read_mapping(numerical_path)
    evaluation = result["evaluation"]
    comparisons = {
        "pytorch_onnx_max_abs_error": "max_abs_error",
        "onnxruntime_p99_ms": "p99_inference_ms",
        "real_test_observation_count": "real_test_observation_count",
        "real_observation_source_sha256": "real_observation_source_sha256",
    }
    for evaluation_key, numerical_key in comparisons.items():
        if evaluation.get(evaluation_key) != numerical.get(numerical_key):
            raise ValueError(
                f"评估与 ONNX 校验报告不一致: {evaluation_key}"
            )
    expected_checksum = f"{sha256_file(archive_path)}  car_rl_model.zip\n"
    if checksum_path.read_text(encoding="ascii") != expected_checksum:
        raise ValueError("car_rl_model.zip.sha256 与压缩包不一致")
    archived_names = {
        "bundle.yaml",
        "controller/car_rl_model.onnx",
        "controller/evaluation.json",
        "controller/metadata.yaml",
        "environment-lock.json",
        "onnx_validation.json",
    }
    with zipfile.ZipFile(archive_path) as stream:
        if set(stream.namelist()) != archived_names:
            raise ValueError("car_rl_model.zip 文件清单不符合生产契约")
        corrupt = stream.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP 校验失败: {corrupt}")
        for name in archived_names:
            archived_digest = hashlib.sha256(stream.read(name)).hexdigest()
            if archived_digest != sha256_file(bundle / name):
                raise ValueError(f"ZIP 与发布目录内容不一致: {name}")


def main() -> None:
    """完成全部生产门禁后事务式发布固定名称模型。"""

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "configs" / "export.yaml")
    known, _ = pre.parse_known_args()
    config = load_workflow_config(known.config)
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--checkpoint", default=config.get("checkpoint", "latest"))
    parser.add_argument("--seed", type=int, default=config.get("seed"))
    parser.add_argument("--test-observations", default=config.get("test_observations"))
    parser.add_argument("--observation-manifest", default=config.get("observation_manifest"))
    parser.add_argument("--evaluation", default=config.get("evaluation"))
    parser.add_argument("--calibration", default=config.get("calibration"))
    parser.add_argument("--isaaclab-root", default=config.get("isaaclab_root"))
    parser.add_argument("--output-root", default=config.get("output_root", "exports"))
    parser.add_argument("--version", default=config.get("version"))
    args = parser.parse_args()
    checkpoint, selected_evaluation = resolve_selection(args.checkpoint)
    actor, metadata, policy_name = load_actor(checkpoint)
    embedded_calibration = select_calibration(None, None, metadata)
    bundle_calibration = (
        production_calibration(embedded_calibration)
        if embedded_calibration else None
    )
    if args.calibration:
        calibration_candidate = Path(str(args.calibration)).expanduser()
        calibration_candidate = (
            calibration_candidate if calibration_candidate.is_absolute()
            else ROOT / calibration_candidate
        ).resolve()
        if calibration_candidate.is_file():
            bundle_calibration = verify_production_overlay(
                read_mapping(calibration_candidate), embedded_calibration
            )
        else:
            selected = select_calibration(args.calibration, None, metadata)
            bundle_calibration = production_calibration(selected)
    seed = args.seed if args.seed is not None else metadata.get("seed")
    if seed is None:
        raise ValueError("seed is missing from checkpoint and export configuration")
    version = args.version or f"{date.today():%Y%m%d}-{policy_name}-seed{seed}"
    if bundle_calibration is None:
        raise ValueError("生产导出需要 checkpoint 内嵌的完整标定")
    evaluation_path = resolve_path(args.evaluation, must_exist=True) if args.evaluation else selected_evaluation
    if evaluation_path is None:
        raise ValueError("生产导出需要 07_evaluate.py 生成的 suite_summary.json")
    if not args.test_observations or not args.observation_manifest:
        raise ValueError("生产导出需要真实观测 NPZ 和 manifest")
    observation_path = resolve_path(args.test_observations, must_exist=True)
    manifest_path = resolve_path(args.observation_manifest, must_exist=True)
    observations, real_count, source_digest = validation_observations(observation_path, manifest_path)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "car_rl_model"
    backup = output_root / ".car_rl_model.previous"
    if backup.exists() and not target.exists():
        os.replace(backup, target)
    elif backup.exists():
        shutil.rmtree(backup)
    staging_root = Path(tempfile.mkdtemp(prefix=".car_rl_model.pending-", dir=output_root))
    work = staging_root / "work"
    work.mkdir()
    onnx_path = export_controller(actor, work / "car_rl_model.onnx")
    numerical = validate_onnx_numerics(actor, onnx_path, observations)
    numerical["real_test_observation_count"] = real_count
    numerical["real_observation_source_sha256"] = source_digest
    asset_manifest = read_mapping(ROOT / "assets" / "asset_manifest.json")
    config_files = list((ROOT / "configs").glob("*.yaml")) + [ROOT / "assets" / "asset_manifest.json"]
    environment = capture_environment_lock(
        ROOT,
        discover_isaaclab_root(args.isaaclab_root),
        asset_manifest["source_fingerprint_sha256"],
        config_files,
    )
    lock_path = work / "environment-lock.json"
    write_environment_lock(lock_path, environment)
    evaluation = portable_project_value(read_mapping(evaluation_path))
    evaluation.update(
        {
            "pytorch_onnx_max_abs_error": numerical["max_abs_error"],
            "onnxruntime_p99_ms": numerical["p99_inference_ms"],
            "onnxruntime_benchmark_platform": platform.platform(),
            "real_test_observation_count": real_count,
            "real_observation_source_sha256": source_digest,
        }
    )
    bundle = staging_root / "published"
    published = False
    try:
        create_bundle(
            onnx_path,
            evaluation,
            environment,
            bundle_calibration,
            bundle,
            version,
        )
        (bundle / "onnx_validation.json").write_text(
            json.dumps(numerical, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        archive, checksum = archive_bundle(
            bundle, staging_root / "car_rl_model.zip"
        )
        os.replace(archive, bundle / "car_rl_model.zip")
        os.replace(checksum, bundle / "car_rl_model.zip.sha256")
        validate_release_directory(bundle)
        if target.exists():
            if not target.is_dir() or target.is_symlink():
                raise FileExistsError(f"导出目标不是普通目录: {target}")
            os.replace(target, backup)
        os.replace(bundle, target)
        published = True
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    if not published:
        raise RuntimeError("生产模型未发布")
    print(f"bundle={target}")
    print(f"onnx={target / 'controller' / 'car_rl_model.onnx'}")
    print(f"archive={target / 'car_rl_model.zip'}")
    print(f"checksum={target / 'car_rl_model.zip.sha256'}")
    print(json.dumps(numerical, indent=2))


if __name__ == "__main__":
    main()
