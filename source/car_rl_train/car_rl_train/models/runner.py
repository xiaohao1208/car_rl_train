"""无需 monkey-patch 上游包即可解析项目策略类的 RSL-RL runner。"""

from __future__ import annotations

import inspect
import hashlib
import warnings
from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from .actor_critic import CarActorCritic


class CarPPO(PPO):
    """在官方 PPO 上增加可恢复的单调学习率衰减。"""

    def __init__(
        self,
        policy,
        *,
        lr_decay_half_life: float = 50.0,
        min_learning_rate: float = 1.0e-5,
        symmetry_contract: str = "car_mirror_v1",
        **kwargs,
    ):
        if lr_decay_half_life <= 0.0:
            raise ValueError("lr_decay_half_life must be positive")
        initial_learning_rate = float(kwargs.get("learning_rate", 1.0e-3))
        if not 0.0 < min_learning_rate <= initial_learning_rate:
            raise ValueError("min_learning_rate must be in (0, learning_rate]")
        if symmetry_contract != "car_mirror_v1":
            raise ValueError("unsupported symmetry contract")
        super().__init__(policy, **kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = float(min_learning_rate)
        self.lr_decay_half_life = float(lr_decay_half_life)
        self._update_step = 0

    def set_update_step(self, step: int) -> None:
        """从 runner 迭代数恢复衰减位置并同步优化器。"""

        if step < 0:
            raise ValueError("update step cannot be negative")
        self._update_step = int(step)
        self.learning_rate = max(
            self.min_learning_rate,
            self.initial_learning_rate
            * 0.5 ** (self._update_step / self.lr_decay_half_life),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self) -> dict[str, float]:
        """按当前全局迭代设置学习率，然后执行官方 PPO 更新。"""

        self.set_update_step(self._update_step)
        losses = super().update()
        self._update_step += 1
        return losses


class CarOnPolicyRunner(OnPolicyRunner):
    """使用项目自有 CarActorCritic、复用官方 PPO 的训练 runner。"""

    def __init__(
        self,
        env,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        """保存可复现元数据，再交给上游 runner 初始化训练状态。"""

        self.checkpoint_metadata = checkpoint_metadata or {}
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def _construct_algorithm(self, obs):
        """解析观测组并创建兼容已安装 RSL-RL 3.1.2 的 PPO。"""

        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn("empirical_normalization is ignored; use policy normalization fields", DeprecationWarning)
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        actor_critic = CarActorCritic(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            **policy_cfg,
        ).to(self.device)
        algorithm_cfg = dict(self.alg_cfg)
        algorithm_cfg.pop("class_name", None)
        supported = set(inspect.signature(PPO.__init__).parameters) - {"self", "policy"}
        supported.update(
            {"lr_decay_half_life", "min_learning_rate", "symmetry_contract"}
        )
        ignored = sorted(set(algorithm_cfg) - supported)
        if ignored:
            warnings.warn(
                "Ignoring Isaac Lab PPO options unsupported by installed RSL-RL "
                f"3.1.2: {', '.join(ignored)}",
                RuntimeWarning,
            )
        algorithm_cfg = {key: value for key, value in algorithm_cfg.items() if key in supported}
        algorithm = CarPPO(
            actor_critic,
            device=self.device,
            **algorithm_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """让新训练和 checkpoint 续训使用同一个全局衰减位置。"""

        self.alg.set_update_step(int(self.current_learning_iteration))
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos=None) -> None:
        """把模型构造与契约元数据嵌入每个新 checkpoint。"""

        metadata = dict(self.checkpoint_metadata)
        policy = self.alg.policy
        digest = hashlib.sha256()
        for name, tensor in sorted(policy.actor.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        raw_noise_parameter = policy.noise_logit.detach()
        noise_std = policy.min_noise_std + (
            policy.max_noise_std - policy.min_noise_std
        ) * raw_noise_parameter.sigmoid()
        if not noise_std.isfinite().all():
            raise RuntimeError("refusing to save policy with non-finite noise std")
        raw_env = self.env.unwrapped

        def scalar(name: str, default: float = 0.0) -> float:
            value = getattr(raw_env, name, None)
            return float(value.item()) if value is not None else default

        termination_values = getattr(
            raw_env, "_car_curriculum_last_termination_rates", None
        )
        termination_names = (
            "time_out",
            "goal_reached",
            "collision",
            "path_deviation",
            "stalled",
            "invalid_input",
        )
        termination_rates = (
            {
                name: float(termination_values[index].item())
                for index, name in enumerate(termination_names)
            }
            if termination_values is not None
            else {name: 0.0 for name in termination_names}
        )
        metadata["training_diagnostics"] = {
            "actor_sha256": digest.hexdigest(),
            "noise_std": [float(value) for value in noise_std.cpu().tolist()],
            "raw_noise_parameter": [
                float(value) for value in raw_noise_parameter.cpu().tolist()
            ],
            "learning_rate": float(self.alg.learning_rate),
            "iteration": int(self.current_learning_iteration),
            "curriculum": {
                "stage": int(scalar("_car_curriculum_stage")),
                "stage_episode_total": int(
                    scalar("_car_curriculum_stage_episode_total")
                ),
                "window_episodes": int(scalar("_car_curriculum_episodes")),
                "window_success_rate": scalar(
                    "_car_curriculum_last_success_rate"
                ),
                "window_collision_rate": scalar(
                    "_car_curriculum_last_collision_rate"
                ),
                "window_static_collision_rate": scalar(
                    "_car_curriculum_last_static_collision_rate"
                ),
                "window_dynamic_collision_rate": scalar(
                    "_car_curriculum_last_dynamic_collision_rate"
                ),
                "termination_rates": termination_rates,
            },
        }
        payload_infos = {"car_rl": metadata}
        if infos is not None:
            payload_infos["upstream"] = infos
        super().save(path, infos=payload_infos)
