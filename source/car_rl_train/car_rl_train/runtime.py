"""训练、播放、评估和诊断共享的环境配置入口。"""

from __future__ import annotations

from typing import Any

from .calibration import apply_calibration, select_calibration
from .tasks.local_navigation.curriculum_config import (
    apply_curriculum_config,
    load_curriculum_config,
)
from .workflow import resolve_path


def configure_environment(
    env_cfg: Any,
    workflow: dict[str, Any],
    *,
    training: bool,
    calibration: str | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """注入唯一课程和标定来源，并返回用于审计的解析结果。"""

    curriculum_path = resolve_path(
        workflow.get("curriculum", "configs/curriculum.yaml"), must_exist=True
    )
    curriculum = load_curriculum_config(curriculum_path)
    apply_curriculum_config(env_cfg, curriculum, evaluation=not training)
    selected_calibration = select_calibration(
        calibration, workflow.get("calibration"), checkpoint_metadata
    )
    physics = apply_calibration(
        env_cfg, selected_calibration, training=training
    )
    return curriculum, selected_calibration, physics
