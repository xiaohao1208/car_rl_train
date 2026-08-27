"""由真实回合终止结果驱动的通用多阶段课程状态机。"""

from __future__ import annotations

import torch


TERMINATION_NAMES = (
    "time_out",
    "goal_reached",
    "collision",
    "path_deviation",
    "stalled",
    "invalid_input",
)


def _initialize(
    env, initial_stage: int = 0, initial_stage_episode_total: int = 0
) -> None:
    env._car_curriculum_stage = torch.tensor(initial_stage, device=env.device, dtype=torch.long)
    env._car_curriculum_passes = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_episodes = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_successes = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_collisions = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_static_collisions = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_dynamic_collisions = torch.tensor(0, device=env.device, dtype=torch.long)
    env._car_curriculum_last_success_rate = torch.tensor(0.0, device=env.device)
    env._car_curriculum_last_collision_rate = torch.tensor(0.0, device=env.device)
    env._car_curriculum_last_static_collision_rate = torch.tensor(0.0, device=env.device)
    env._car_curriculum_last_dynamic_collision_rate = torch.tensor(0.0, device=env.device)
    env._car_curriculum_stage_episode_total = torch.tensor(
        initial_stage_episode_total, device=env.device, dtype=torch.long
    )
    env._car_curriculum_termination_counts = torch.zeros(
        len(TERMINATION_NAMES), device=env.device, dtype=torch.long
    )
    env._car_curriculum_last_termination_rates = torch.zeros(
        len(TERMINATION_NAMES), device=env.device
    )


def curriculum_stage(
    env,
    env_ids,
    stages: list[dict],
    window_episodes: int = 500,
    consecutive_passes: int = 3,
):
    """按已完成回合的成功率/碰撞率窗口推进课程阶段。"""

    if not hasattr(env, "_car_curriculum_stage"):
        command_cfg = env.command_manager.get_term("navigation").cfg
        initial = int(command_cfg.curriculum_stage)
        resume_total = int(
            getattr(command_cfg, "resume_stage_episode_total", 0)
        )
        if resume_total < 0:
            raise ValueError("resume stage episode total cannot be negative")
        _initialize(env, initial, resume_total)
    command = env.command_manager.get_term("navigation")
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel():
        terms = {
            name: env.termination_manager.get_term(name)[ids]
            for name in TERMINATION_NAMES
        }
        completed = torch.zeros_like(terms["goal_reached"])
        for value in terms.values():
            completed |= value
        current_stage = int(env._car_curriculum_stage.item())
        current_stage_episode = command.episode_stage[ids] == current_stage
        eligible = completed & current_stage_episode
        if eligible.any():
            count = eligible.sum()
            env._car_curriculum_episodes += count
            env._car_curriculum_stage_episode_total += count
            env._car_curriculum_successes += (terms["goal_reached"] & eligible).sum()
            env._car_curriculum_collisions += (terms["collision"] & eligible).sum()
            static_source = getattr(env, "_car_static_collision", None)
            dynamic_source = getattr(env, "_car_dynamic_collision", None)
            static_collision = (
                static_source[ids] if static_source is not None else terms["collision"]
            )
            dynamic_collision = (
                dynamic_source[ids]
                if dynamic_source is not None
                else torch.zeros_like(terms["collision"])
            )
            env._car_curriculum_static_collisions += (static_collision & eligible).sum()
            env._car_curriculum_dynamic_collisions += (dynamic_collision & eligible).sum()
            for index, name in enumerate(TERMINATION_NAMES):
                env._car_curriculum_termination_counts[index] += (
                    terms[name] & eligible
                ).sum()

    if int(env._car_curriculum_episodes.item()) >= int(window_episodes):
        total = float(env._car_curriculum_episodes.item())
        success_rate = float(env._car_curriculum_successes.item()) / total
        collision_rate = float(env._car_curriculum_collisions.item()) / total
        env._car_curriculum_last_success_rate.fill_(success_rate)
        env._car_curriculum_last_collision_rate.fill_(collision_rate)
        env._car_curriculum_last_static_collision_rate.fill_(
            float(env._car_curriculum_static_collisions.item()) / total
        )
        env._car_curriculum_last_dynamic_collision_rate.fill_(
            float(env._car_curriculum_dynamic_collisions.item()) / total
        )
        env._car_curriculum_last_termination_rates.copy_(
            env._car_curriculum_termination_counts.float() / total
        )
        stage_index = int(env._car_curriculum_stage.item())
        stage_cfg = stages[stage_index]
        passed = (
            success_rate >= float(stage_cfg["success_rate"])
            and collision_rate <= float(stage_cfg["max_collision_rate"])
            and int(env._car_curriculum_stage_episode_total.item())
            >= int(stage_cfg["min_stage_episodes"])
        )
        if passed:
            env._car_curriculum_passes += 1
        else:
            env._car_curriculum_passes.zero_()
        if (
            int(env._car_curriculum_passes.item()) >= int(consecutive_passes)
            and stage_index < len(stages) - 1
        ):
            env._car_curriculum_stage += 1
            env._car_curriculum_passes.zero_()
            command.cfg.curriculum_stage = int(env._car_curriculum_stage.item())
            if hasattr(command.cfg, "resume_stage_episode_total"):
                command.cfg.resume_stage_episode_total = 0
            env._car_curriculum_stage_episode_total.zero_()
            env._car_curriculum_last_success_rate.zero_()
            env._car_curriculum_last_collision_rate.zero_()
            env._car_curriculum_last_static_collision_rate.zero_()
            env._car_curriculum_last_dynamic_collision_rate.zero_()
            env._car_curriculum_last_termination_rates.zero_()
        env._car_curriculum_episodes.zero_()
        env._car_curriculum_successes.zero_()
        env._car_curriculum_collisions.zero_()
        env._car_curriculum_static_collisions.zero_()
        env._car_curriculum_dynamic_collisions.zero_()
        env._car_curriculum_termination_counts.zero_()

    return {
        "stage": float(env._car_curriculum_stage.item()),
        "window_success_rate": float(env._car_curriculum_last_success_rate.item()),
        "window_collision_rate": float(env._car_curriculum_last_collision_rate.item()),
        "window_static_collision_rate": float(
            env._car_curriculum_last_static_collision_rate.item()
        ),
        "window_dynamic_collision_rate": float(
            env._car_curriculum_last_dynamic_collision_rate.item()
        ),
        "consecutive_passes": float(env._car_curriculum_passes.item()),
        "window_episodes": float(env._car_curriculum_episodes.item()),
        "stage_episode_total": float(
            env._car_curriculum_stage_episode_total.item()
        ),
        **{
            f"window_{name}_rate": float(
                env._car_curriculum_last_termination_rates[index].item()
            )
            for index, name in enumerate(TERMINATION_NAMES)
        },
    }
