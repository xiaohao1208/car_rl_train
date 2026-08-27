"""实车标定驱动的整车质量事件。"""

from __future__ import annotations

import math


BODY_MASS_WEIGHTS = {
    "base_link": 0.825,
    "left_wheel_link": 0.050,
    "right_wheel_link": 0.050,
    "imu_link": 0.00625,
    "laser_stand_link": 0.0125,
    "laser_link": 0.03125,
    "front_caster_link": 0.0125,
    "back_caster_link": 0.0125,
}
# 未列出的附属刚体仍分到一个极小但对 PhysX 安全的正质量。
MIN_AUXILIARY_WEIGHT = 1.0e-5


def normalized_body_mass_weights(body_names: list[str]) -> list[float]:
    """返回与 articulation body 顺序一致且和为 1 的质量权重。"""

    missing = sorted(set(BODY_MASS_WEIGHTS) - set(body_names))
    if missing:
        raise ValueError(f"robot asset misses calibrated mass bodies: {missing}")
    raw = [BODY_MASS_WEIGHTS.get(name, MIN_AUXILIARY_WEIGHT) for name in body_names]
    total = math.fsum(raw)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("body mass weights must have a finite positive sum")
    return [value / total for value in raw]


def randomize_total_mass(
    env,
    env_ids,
    asset_cfg,
    total_mass_range_kg: tuple[float, float],
):
    """为每个环境采样整车总质量，并按部件权重同步质量与惯量。"""

    import torch

    low, high = (float(value) for value in total_mass_range_kg)
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
        raise ValueError("total_mass_range_kg must be finite, positive, and ordered")
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")
    else:
        env_ids = env_ids.to(device="cpu", dtype=torch.long)
    if len(env_ids) == 0:
        return

    masses = asset.root_physx_view.get_masses()
    inertias = asset.root_physx_view.get_inertias()
    default_mass = asset.data.default_mass.to(device="cpu", dtype=masses.dtype)
    default_inertia = asset.data.default_inertia.to(
        device="cpu", dtype=inertias.dtype
    )
    if torch.any(default_mass[env_ids, :] <= 0.0):
        raise ValueError("default body masses must all be positive")
    weights = torch.tensor(
        normalized_body_mass_weights(list(asset.body_names)),
        dtype=masses.dtype,
        device="cpu",
    )
    sampled_totals = torch.empty(
        len(env_ids), dtype=masses.dtype, device="cpu"
    ).uniform_(low, high)
    body_masses = sampled_totals[:, None] * weights[None, :]
    masses[env_ids, :] = body_masses
    asset.root_physx_view.set_masses(masses, env_ids)

    ratios = body_masses / default_mass[env_ids, :]
    inertias[env_ids, :, :] = default_inertia[env_ids, :, :] * ratios[..., None]
    asset.root_physx_view.set_inertias(inertias, env_ids)
    env._car_sampled_total_mass_kg = sampled_totals.clone()
