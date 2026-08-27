"""[必选] 把部署端记录的真实 86 维观测转换为生产验证资料。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from car_rl_train.workflow import ROOT


OBSERVATION_CONTRACT_VERSION = 1
OBSERVATION_SIZE = 86
MINIMUM_COUNT = 4000
MINIMUM_UNIQUE_COUNT = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("观测元数据必须是YAML mapping")
    expected = {
        "schema_version": 1,
        "source": "real_car_raw",
        "completed": True,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"观测元数据{name}无效: {value.get(name)!r} != {expected_value!r}"
            )
    return value


def load_observations(path: Path) -> np.ndarray:
    expected_header = [f"observation_{index}" for index in range(OBSERVATION_SIZE)]
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("真实观测CSV为空") from error
        if header != expected_header:
            raise ValueError("真实观测CSV列名或86维顺序不符合contract v1")
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != OBSERVATION_SIZE:
                raise ValueError(f"真实观测CSV第{line_number}行不是86列")
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise ValueError(f"真实观测CSV第{line_number}行包含非数值") from error
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (len(rows), OBSERVATION_SIZE):
        raise ValueError("真实观测数组shape无效")
    if values.shape[0] < MINIMUM_COUNT:
        raise ValueError(f"真实观测至少需要{MINIMUM_COUNT}条，当前={values.shape[0]}")
    if not np.all(np.isfinite(values)):
        raise ValueError("真实观测包含NaN或Inf")
    if np.any(values < -1.000001) or np.any(values > 1.000001):
        raise ValueError("真实观测超出contract v1归一化范围[-1,1]")
    unique_count = int(np.unique(values, axis=0).shape[0])
    if unique_count < MINIMUM_UNIQUE_COUNT:
        raise ValueError(
            "真实观测多样性不足，至少需要"
            f"{MINIMUM_UNIQUE_COUNT}条不同观测，当前={unique_count}"
        )
    return values


def import_observations(
    csv_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> tuple[Path, Path]:
    csv_path = csv_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("真实观测CSV或元数据不存在")
    if output_path.suffix != ".npz":
        raise ValueError("真实观测输出必须使用.npz扩展名")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("真实观测NPZ或manifest已存在，拒绝覆盖")

    metadata = load_metadata(metadata_path)
    csv_digest = sha256_file(csv_path)
    if metadata.get("csv_sha256") != csv_digest:
        raise ValueError("真实观测CSV摘要与记录器元数据不一致")
    observations = load_observations(csv_path)
    if int(metadata.get("count", -1)) != int(observations.shape[0]):
        raise ValueError("真实观测数量与记录器元数据不一致")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".pending", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, observation=observations)
            stream.flush()
            os.fsync(stream.fileno())
        npz_digest = sha256_file(temporary)
        manifest = {
            "schema_version": 1,
            "source": "real_car",
            "production_eligible": True,
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "count": int(observations.shape[0]),
            "unique_count": int(np.unique(observations, axis=0).shape[0]),
            "shape": [int(observations.shape[0]), OBSERVATION_SIZE],
            "sha256": npz_digest,
            "raw_csv_sha256": csv_digest,
        }
        manifest_temporary = Path(str(manifest_path) + ".pending")
        if manifest_temporary.exists():
            raise FileExistsError(f"manifest pending文件已存在: {manifest_temporary}")
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        os.replace(manifest_temporary, manifest_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        Path(str(manifest_path) + ".pending").unlink(missing_ok=True)
        raise
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "observations" / "real_observations.csv",
        help="记录器生成的 CSV",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "observations" / "real_observations.metadata.yaml",
        help="记录器元数据 YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "observations" / "real_observations.npz",
        help="目标 NPZ",
    )
    args = parser.parse_args()
    output, manifest = import_observations(args.input, args.metadata, args.output)
    print(f"observations={output}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
