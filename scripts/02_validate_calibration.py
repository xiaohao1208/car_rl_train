"""[必选] 严格重算并校验项目内的真实小车标定文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from car_rl_train.calibration import load_calibration
from car_rl_train.workflow import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration")
    args = parser.parse_args()
    train_config = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(train_config, dict) or train_config.get("calibration") != "calibration":
        raise ValueError("configs/train.yaml 必须固定使用项目 calibration 目录")
    calibration = load_calibration(args.calibration)
    if calibration is None:
        raise RuntimeError("标定目录未能加载")
    manual = calibration.get("manual_fields") or {}
    if manual.get("total_mass_kg") is None or manual.get("wheel_friction_center") is None:
        raise ValueError("标定缺少 total_mass_kg 或 wheel_friction_center")
    print(json.dumps(calibration, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
