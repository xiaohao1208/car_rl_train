"""兼容 RSL-RL 3.1.2、确定性均值有界的 actor-critic。"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.distributions import Normal

from .split_fusion import SplitFusionEncoder


def _export_safe_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """计算可由 ONNX 稳定复现的 ``atan2``，并定义零向量方向为零。"""

    # PyTorch 的旧式 ONNX 导出器会把 atan2 展开为除法和条件选择。
    # y == 0 时，该展开式会在 (0, 0) 产生 NaN，也会在负 x 轴丢失
    # 象限符号。先让未选中的 atan2 分支远离零除法，再显式定义轴线，
    # 可同时保持正常观测的原始行为和异常观测的有限安全输出。
    zero_y = y == 0.0
    safe_y = torch.where(zero_y, torch.full_like(y, 1.0e-6), y)
    general = torch.atan2(safe_y, x)
    axial = torch.where(
        x < 0.0,
        torch.full_like(x, math.pi),
        torch.zeros_like(x),
    )
    return torch.where(zero_y, axial, general)


class _Actor(torch.nn.Module):
    """由独立编码器、线速度残差和确定性转向控制器组成的 actor。"""

    def __init__(self, encoder: torch.nn.Module, feature_dim: int):
        """初始化 actor 编码器与唯一可学习的线速度残差头。"""

        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(feature_dim, 1)
        torch.nn.init.orthogonal_(self.head.weight, gain=0.01)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """输出带可学习残差制动先验、严格位于 [-1,1] 的动作均值。"""

        learned_linear = torch.tanh(self.head(self.encoder(observation)))[:, 0]
        # 契约索引 78:80 是除以 3 m 的车体坐标系目标向量。车辆在
        # 0.20 m 内逐渐把线速度先验由 0.05 m/s 收紧到接近零，避免
        # PPO 必须仅靠稀疏终止回报学会制动。0.20--0.25 m 的混合带
        # 保证切换连续；进入 0.20 m 后固定进站线速度，防止残差漂移。
        goal_distance_m = torch.linalg.vector_norm(
            observation[:, 78:80], dim=-1
        ) * 3.0
        # 雷达顺序是 [-pi, pi)，正前方位于索引 36。速度包络只考虑
        # 前方约 +/-30 度；侧向静态净空不应让车辆在合法路径上停车。
        front_clearance_m = (
            torch.amin(observation[:, 30:43], dim=-1) * 5.95 + 0.05
        )
        path_heading = _export_safe_atan2(
            observation[:, 73], observation[:, 72]
        )
        # 固定曲率速度包络：直线路径约 0.095 m/s，前视方向偏转达到
        # 30 度时降至 0.07 m/s。网络只允许提供 5% 的有界微调，不能
        # 在长时间 PPO 更新后把已验证的几何控制器推入过快或爬行状态。
        turn_scale = torch.clamp(
            torch.abs(torch.sin(path_heading)) / 0.5, min=0.0, max=1.0
        )
        cruise_prior = 0.90 - 0.50 * turn_scale
        cruise_linear = torch.clamp(
            cruise_prior + 0.05 * learned_linear, min=0.05, max=0.95
        )
        braking_prior = -torch.clamp(
            (0.20 - goal_distance_m) / (0.20 - 0.08),
            min=0.0,
            max=0.70,
        )
        braking_blend = torch.clamp(
            (0.25 - goal_distance_m) / (0.25 - 0.20),
            min=0.0,
            max=1.0,
        )
        linear = (
            (1.0 - braking_blend) * cruise_linear
            + braking_blend * braking_prior
        )
        yaw_error = _export_safe_atan2(
            observation[:, 80], observation[:, 81]
        )
        # 近目标固定进站曲线不能覆盖紧急避障。考虑约 0.18 s 实车动作
        # 延迟和移动障碍相对速度，从 0.30 m 起连续减速，到 0.20 m 停车；
        # 远处只是限速而非急停，避免窄通道的侧前方障碍触发停滞。
        emergency_blend = torch.clamp(
            (0.30 - front_clearance_m) / (0.30 - 0.20),
            min=0.0,
            max=1.0,
        )
        linear = (1.0 - emergency_blend) * linear - emergency_blend

        # 0.3 m 前视点与目标偏航构成确定性的纯追踪转向。长路径会把
        # 极小的学习偏置持续累积，因此转向不再包含不可审计的网络残差。
        linear_speed_mps = 0.05 * (linear + 1.0)
        path_angular = torch.clamp(
            2.0 * linear_speed_mps * torch.sin(path_heading) / 0.30,
            min=-1.0,
            max=1.0,
        )
        goal_angular = torch.clamp(1.5 * yaw_error, min=-1.0, max=1.0)
        # 先完整追踪到目标位置，再在位置容差内部停车并原地对齐朝向。
        # 0.07--0.04 m 的连续带对观测偏差留出余量，并满足 0.08 m
        # 真实位置生产门禁，避免在容差外提前停车。
        pose_blend = torch.clamp(
            (0.07 - goal_distance_m) / (0.07 - 0.04),
            min=0.0,
            max=1.0,
        )
        linear = (1.0 - pose_blend) * linear - pose_blend
        steering_prior = (
            (1.0 - pose_blend) * path_angular + pose_blend * goal_angular
        )
        return torch.stack((linear, steering_prior), dim=-1)


class _Critic(torch.nn.Module):
    """由独立编码器和标量价值头组成的 critic。"""

    def __init__(self, encoder: torch.nn.Module, feature_dim: int):
        """初始化 critic 编码器与状态价值预测头。"""

        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(feature_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """预测每条观测对应的标量状态价值。"""

        return self.head(self.encoder(observation))


class CarActorCritic(torch.nn.Module):
    """Actor 与 critic 分别使用所选编码器的独立实例。"""

    is_recurrent = False

    def __init__(
        self,
        obs: Any | None = None,
        obs_groups: dict[str, list[str]] | None = None,
        num_actions: int = 2,
        observation_dim: int = 86,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.6,
        noise_std_type: str = "bounded_logit",
        min_noise_std: float = 0.05,
        max_noise_std: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **_: Any,
    ):
        """按 RSL-RL 3.1.2 接口创建策略、价值网络和探索噪声。"""

        super().__init__()
        if num_actions != 2:
            raise ValueError("controller action must have two elements")
        self.obs_groups = obs_groups or {"policy": ["policy"], "critic": ["policy"]}
        if obs is not None:
            actor_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups["policy"])
            critic_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups["critic"])
            if actor_dim != 86 or critic_dim != 86:
                raise ValueError(f"policy and critic observations must both be 86, got {actor_dim}/{critic_dim}")
            observation_dim = actor_dim
        if noise_std_type != "bounded_logit":
            raise ValueError("only bounded_logit exploration noise is supported")
        actor_encoder = SplitFusionEncoder(
            observation_dim=observation_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        critic_encoder = SplitFusionEncoder(
            observation_dim=observation_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        self.actor = _Actor(actor_encoder, actor_encoder.output_dim)
        self.critic = _Critic(critic_encoder, critic_encoder.output_dim)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("contract observations are already normalized; empirical normalization is disabled")
        self.actor_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalizer = torch.nn.Identity()
        self.noise_std_type = noise_std_type
        if not 0.0 < min_noise_std <= init_noise_std <= max_noise_std:
            raise ValueError("noise std must satisfy 0 < min <= init <= max")
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        fraction = (init_noise_std - self.min_noise_std) / (
            self.max_noise_std - self.min_noise_std
        )
        initial_logit = math.log(fraction / (1.0 - fraction))
        self.noise_logit = torch.nn.Parameter(
            initial_logit * torch.ones(num_actions)
        )
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """直接执行确定性 actor，便于常规 PyTorch 调用。"""

        return self.actor(observation)

    def export_actor(self) -> torch.nn.Module:
        """返回不含 critic 与随机采样的确定性部署模块。"""

        return self.actor

    def _concatenate(self, obs: Any, group: str) -> torch.Tensor:
        """按 RSL-RL 观测组声明拼接张量。"""

        if isinstance(obs, torch.Tensor):
            return obs
        return torch.cat([obs[name] for name in self.obs_groups[group]], dim=-1)

    def get_actor_obs(self, obs: Any) -> torch.Tensor:
        """提取 actor 使用的 policy 观测。"""

        return self._concatenate(obs, "policy")

    def get_critic_obs(self, obs: Any) -> torch.Tensor:
        """提取 critic 使用的观测。"""

        return self._concatenate(obs, "critic")

    def update_distribution(self, observation: torch.Tensor) -> None:
        """用当前动作均值与可学习标准差更新高斯策略分布。"""

        mean = self.actor(observation)
        std = self.min_noise_std + (
            self.max_noise_std - self.min_noise_std
        ) * torch.sigmoid(self.noise_logit)
        std = std.expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError("policy distribution contains NaN/Inf")
        self.distribution = Normal(mean, std)

    def act(self, obs: Any, **_: Any) -> torch.Tensor:
        """训练阶段从随机策略分布采样动作。"""

        observation = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(observation)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: Any) -> torch.Tensor:
        """推理阶段仅返回确定性 actor 输出。"""

        return self.actor(self.actor_obs_normalizer(self.get_actor_obs(obs)))

    def evaluate(self, obs: Any, **_: Any) -> torch.Tensor:
        """计算 critic 状态价值。"""

        return self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs)))

    @property
    def action_mean(self) -> torch.Tensor:
        """返回最近一次策略分布的动作均值。"""

        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """返回最近一次策略分布的动作标准差。"""

        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """返回两维动作分布的联合熵。"""

        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算给定动作在当前策略分布下的对数概率。"""

        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """满足 RSL-RL 接口；前馈网络没有需要重置的隐状态。"""

        del dones

    def update_normalization(self, obs: Any) -> None:
        """满足 RSL-RL 接口；契约观测已归一化。"""

        del obs

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> bool:
        """载入 checkpoint，并按 RSL-RL 约定返回成功标志。"""

        super().load_state_dict(state_dict, strict=strict)
        return True
