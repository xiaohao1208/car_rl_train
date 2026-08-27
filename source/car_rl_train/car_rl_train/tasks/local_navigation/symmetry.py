"""差速小车 86 维观测和 2 维动作的左右镜像增强。"""

from __future__ import annotations

import torch
from tensordict import TensorDict


@torch.no_grad()
def mirror_navigation_batch(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """返回原样本和左右镜像样本；env 仅满足 RSL-RL 接口。"""

    del env
    if obs is not None:
        batch_size = obs.batch_size[0]
        augmented_obs = obs.repeat(2)
        policy = obs["policy"]
        if policy.shape[-1] != 86:
            raise ValueError("symmetry requires the 86-dimensional controller contract")
        mirrored = policy.clone()
        lidar_indices = torch.remainder(
            -torch.arange(72, device=policy.device), 72
        )
        mirrored[..., :72] = policy[..., lidar_indices]
        for index in (73, 75, 77, 79, 80, 83, 85):
            mirrored[..., index] = -policy[..., index]
        augmented_obs["policy"][:batch_size] = policy
        augmented_obs["policy"][batch_size:] = mirrored
    else:
        augmented_obs = None

    if actions is not None:
        if actions.shape[-1] != 2:
            raise ValueError("symmetry requires two controller actions")
        batch_size = actions.shape[0]
        mirrored_actions = actions.clone()
        mirrored_actions[..., 1] = -actions[..., 1]
        augmented_actions = torch.cat((actions, mirrored_actions), dim=0)
        if augmented_actions.shape[0] != 2 * batch_size:
            raise RuntimeError("invalid mirrored action batch")
    else:
        augmented_actions = None
    return augmented_obs, augmented_actions
