"""可在 Isaac Sim 外安全导入的 Gymnasium 任务注册入口。"""

from __future__ import annotations

import gymnasium as gym

_REGISTERED = False


def register_tasks() -> None:
    """注册训练与 Play 两个 Gym 环境，重复导入时保持幂等。"""

    global _REGISTERED
    if _REGISTERED:
        return
    registrations = {
        "Car-LocalNavigation-v0": "CarLocalNavigationEnvCfg",
        "Car-LocalNavigation-Play-v0": "CarLocalNavigationPlayEnvCfg",
        "Car-LocalNavigation-Eval-v0": "CarLocalNavigationEvaluationEnvCfg",
    }
    for task_id, cfg_name in registrations.items():
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": (
                        "car_rl_train.tasks.local_navigation.env_cfg:" + cfg_name
                    ),
                    "rsl_rl_cfg_entry_point": (
                        "car_rl_train.tasks.local_navigation.agents.rsl_rl_ppo_cfg:"
                        "CarNavigationPPORunnerCfg"
                    ),
                },
            )
    _REGISTERED = True


__all__ = ["register_tasks"]
