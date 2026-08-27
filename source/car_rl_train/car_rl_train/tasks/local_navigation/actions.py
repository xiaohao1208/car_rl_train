"""包含延迟、一阶电机响应和加速度限制的差速动作项。"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class DifferentialDriveAction(ActionTerm):
    """把策略动作转换为带动力学约束的左右轮速度目标。"""

    cfg: "DifferentialDriveActionCfg"

    def __init__(self, cfg: "DifferentialDriveActionCfg", env):
        """解析轮关节并创建动作延迟与电机响应状态。"""

        super().__init__(cfg, env)
        ranges = (
            (cfg.motor_time_constant_sec_min, cfg.motor_time_constant_sec_max),
            (cfg.left_motor_gain_min, cfg.left_motor_gain_max),
            (cfg.right_motor_gain_min, cfg.right_motor_gain_max),
            (cfg.max_linear_accel_mps2_min, cfg.max_linear_accel_mps2_max),
            (cfg.max_linear_decel_mps2_min, cfg.max_linear_decel_mps2_max),
            (cfg.max_angular_accel_radps2_min, cfg.max_angular_accel_radps2_max),
            (cfg.max_angular_decel_radps2_min, cfg.max_angular_decel_radps2_max),
        )
        if any(float(low) <= 0.0 or float(high) < float(low) for low, high in ranges):
            raise ValueError("dynamics ranges must be positive and ordered")
        if not 0 <= cfg.min_delay_policy_steps <= cfg.max_delay_policy_steps:
            raise ValueError("delay policy-step range is invalid")
        joint_ids, names = self._asset.find_joints(cfg.joint_names)
        if set(names) != {"left_wheel_joint", "right_wheel_joint"}:
            raise ValueError(f"required wheel joints not found: {names}")
        # 不依赖 articulation 内部遍历顺序，始终强制保持左轮、右轮排列。
        mapping = {name: index for index, name in zip(joint_ids, names)}
        self._joint_ids = [mapping["left_wheel_joint"], mapping["right_wheel_joint"]]
        self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.twist_command = torch.zeros_like(self._raw_actions)
        self.wheel_target = torch.zeros_like(self._raw_actions)
        self.wheel_feedback_target = torch.zeros_like(self._raw_actions)
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._queue = torch.zeros(
            (cfg.max_delay_policy_steps + 1, self.num_envs, 2), device=self.device
        )
        self._motor_time_constant = torch.full(
            (self.num_envs,), float(cfg.motor_time_constant_sec), device=self.device
        )
        self._left_motor_gain = torch.full(
            (self.num_envs,), float(cfg.left_motor_gain), device=self.device
        )
        self._right_motor_gain = torch.full(
            (self.num_envs,), float(cfg.right_motor_gain), device=self.device
        )
        self._linear_accel = torch.full(
            (self.num_envs,), float(cfg.max_linear_accel_mps2), device=self.device
        )
        self._linear_decel = torch.full(
            (self.num_envs,), float(cfg.max_linear_decel_mps2), device=self.device
        )
        self._angular_accel = torch.full(
            (self.num_envs,), float(cfg.max_angular_accel_radps2), device=self.device
        )
        self._angular_decel = torch.full(
            (self.num_envs,), float(cfg.max_angular_decel_radps2), device=self.device
        )
        self._traction_scale = torch.ones(self.num_envs, device=self.device)
        self._sampled_drive_mass = torch.ones(self.num_envs, device=self.device)
        self._sampled_drive_friction = torch.ones(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        """返回策略动作维度。"""

        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """返回策略提交的原始动作。"""

        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """返回裁剪到有效范围后的动作。"""

        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """完成裁剪、速度映射、差速换算和延迟队列采样。"""

        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"expected actions [{self.num_envs},2], got {actions.shape}")
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(torch.clamp(actions, -1.0, 1.0))
        linear = 0.5 * self.cfg.max_linear_speed_mps * (self._processed_actions[:, 0] + 1.0)
        angular = self.cfg.max_angular_speed_radps * self._processed_actions[:, 1]
        self.twist_command[:, 0] = linear
        self.twist_command[:, 1] = angular
        left = (linear - angular * self.cfg.wheel_separation_m / 2.0) / self.cfg.wheel_radius_m
        right = (linear + angular * self.cfg.wheel_separation_m / 2.0) / self.cfg.wheel_radius_m
        self._queue = torch.roll(self._queue, shifts=1, dims=0)
        self._queue[0, :, 0] = left * self._left_motor_gain
        self._queue[0, :, 1] = right * self._right_motor_gain
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.wheel_target.copy_(self._queue[self._delay_steps, env_ids])

    def apply_actions(self) -> None:
        """按物理步应用一阶电机响应及轮端加速度限制。"""

        dt = float(self._env.physics_dt)
        alpha = torch.clamp(dt / torch.clamp(self._motor_time_constant, min=dt), max=1.0)
        alpha = alpha.unsqueeze(1)
        desired_delta = alpha * (self.wheel_target - self.wheel_feedback_target)
        # 先变换到底盘 (v,w) 空间分别限制线/角加速度，再变回左右轮；这样
        # 直行和转向的实测门限不会被粗略的单轮上限互相覆盖。
        radius = self.cfg.wheel_radius_m
        separation = self.cfg.wheel_separation_m
        delta_linear = 0.5 * radius * (desired_delta[:, 0] + desired_delta[:, 1])
        delta_angular = radius * (desired_delta[:, 1] - desired_delta[:, 0]) / separation
        current_linear = 0.5 * radius * (
            self.wheel_feedback_target[:, 0] + self.wheel_feedback_target[:, 1]
        )
        current_angular = radius * (
            self.wheel_feedback_target[:, 1] - self.wheel_feedback_target[:, 0]
        ) / separation
        target_linear = 0.5 * radius * (
            self.wheel_target[:, 0] + self.wheel_target[:, 1]
        )
        target_angular = radius * (
            self.wheel_target[:, 1] - self.wheel_target[:, 0]
        ) / separation
        linear_limit = torch.where(
            torch.abs(target_linear) > torch.abs(current_linear),
            self._linear_accel, self._linear_decel,
        ) * self._traction_scale * dt
        angular_limit = torch.where(
            torch.abs(target_angular) > torch.abs(current_angular),
            self._angular_accel, self._angular_decel,
        ) * self._traction_scale * dt
        delta_linear = torch.maximum(
            torch.minimum(delta_linear, linear_limit), -linear_limit
        )
        delta_angular = torch.maximum(
            torch.minimum(delta_angular, angular_limit), -angular_limit
        )
        desired_delta[:, 0] = (
            delta_linear - 0.5 * separation * delta_angular
        ) / radius
        desired_delta[:, 1] = (
            delta_linear + 0.5 * separation * delta_angular
        ) / radius
        self.wheel_feedback_target.add_(desired_delta)
        self._asset.set_joint_velocity_target(
            self.wheel_feedback_target, joint_ids=self._joint_ids
        )
        if self.cfg.enforce_planar_motion:
            # 部署接口是二维 twist。直接把标定后的轮端响应换算为平面根速度，
            # 避免自由 6-DoF 接触中的侧翻/空转，同时保留质量、摩擦对加速度的影响。
            linear = 0.5 * radius * (
                self.wheel_feedback_target[:, 0] + self.wheel_feedback_target[:, 1]
            )
            angular = radius * (
                self.wheel_feedback_target[:, 1] - self.wheel_feedback_target[:, 0]
            ) / separation
            quaternion = self._asset.data.root_quat_w
            w, x, y, z = quaternion.unbind(dim=-1)
            yaw = torch.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            velocity = self._asset.data.root_vel_w.clone()
            velocity[:, 0] = torch.cos(yaw) * linear
            velocity[:, 1] = torch.sin(yaw) * linear
            velocity[:, 2:5] = 0.0
            velocity[:, 5] = angular
            self._asset.write_root_velocity_to_sim(velocity)

    def reset(self, env_ids=None) -> None:
        """清空指定环境的动作状态，并重新采样策略延迟。"""

        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.twist_command[env_ids] = 0.0
        self.wheel_target[env_ids] = 0.0
        self.wheel_feedback_target[env_ids] = 0.0
        self._queue[:, env_ids] = 0.0
        count = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        command = self._env.command_manager.get_term("navigation")
        scales = command.randomization_scales(env_ids)
        mass_range = self._env.cfg.events.total_mass.params["total_mass_range_kg"]
        friction_range = self._env.cfg.events.wheel_material.params[
            "static_friction_range"
        ]
        mass_center = 0.5 * (float(mass_range[0]) + float(mass_range[1]))
        friction_center = 0.5 * (
            float(friction_range[0]) + float(friction_range[1])
        )
        if self.cfg.randomize_dynamics:
            mass_half = 0.5 * (float(mass_range[1]) - float(mass_range[0]))
            friction_half = 0.5 * (
                float(friction_range[1]) - float(friction_range[0])
            )
            sampled_mass = mass_center + (
                torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                * mass_half
                * scales
            )
            sampled_friction = friction_center + (
                torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                * friction_half
                * scales
            )
        else:
            sampled_mass = torch.full((count,), mass_center, device=self.device)
            sampled_friction = torch.full(
                (count,), friction_center, device=self.device
            )
        self._sampled_drive_mass[env_ids] = sampled_mass
        self._sampled_drive_friction[env_ids] = sampled_friction
        self._traction_scale[env_ids] = torch.clamp(
            (sampled_friction / friction_center) * (mass_center / sampled_mass),
            0.5,
            1.5,
        )
        self._env._car_drive_mass_kg = self._sampled_drive_mass
        self._env._car_drive_friction = self._sampled_drive_friction
        if self.cfg.randomize_dynamics:
            for tensor, low, high in (
                (self._motor_time_constant, self.cfg.motor_time_constant_sec_min,
                 self.cfg.motor_time_constant_sec_max),
                (self._left_motor_gain, self.cfg.left_motor_gain_min,
                 self.cfg.left_motor_gain_max),
                (self._right_motor_gain, self.cfg.right_motor_gain_min,
                 self.cfg.right_motor_gain_max),
                (self._linear_accel, self.cfg.max_linear_accel_mps2_min,
                 self.cfg.max_linear_accel_mps2_max),
                (self._linear_decel, self.cfg.max_linear_decel_mps2_min,
                 self.cfg.max_linear_decel_mps2_max),
                (self._angular_accel, self.cfg.max_angular_accel_radps2_min,
                 self.cfg.max_angular_accel_radps2_max),
                (self._angular_decel, self.cfg.max_angular_decel_radps2_min,
                 self.cfg.max_angular_decel_radps2_max),
            ):
                center = 0.5 * (float(low) + float(high))
                half = 0.5 * (float(high) - float(low))
                tensor[env_ids] = center + (
                    torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
                    * half
                    * scales
                )
        else:
            for tensor, low, high in (
                (self._motor_time_constant, self.cfg.motor_time_constant_sec_min,
                 self.cfg.motor_time_constant_sec_max),
                (self._left_motor_gain, self.cfg.left_motor_gain_min,
                 self.cfg.left_motor_gain_max),
                (self._right_motor_gain, self.cfg.right_motor_gain_min,
                 self.cfg.right_motor_gain_max),
                (self._linear_accel, self.cfg.max_linear_accel_mps2_min,
                 self.cfg.max_linear_accel_mps2_max),
                (self._linear_decel, self.cfg.max_linear_decel_mps2_min,
                 self.cfg.max_linear_decel_mps2_max),
                (self._angular_accel, self.cfg.max_angular_accel_radps2_min,
                 self.cfg.max_angular_accel_radps2_max),
                (self._angular_decel, self.cfg.max_angular_decel_radps2_min,
                 self.cfg.max_angular_decel_radps2_max),
            ):
                tensor[env_ids] = 0.5 * (float(low) + float(high))
        if self.cfg.randomize_delay:
            center = int(round(0.5 * (self.cfg.min_delay_policy_steps + self.cfg.max_delay_policy_steps)))
            sampled = torch.randint(
                self.cfg.min_delay_policy_steps,
                self.cfg.max_delay_policy_steps + 1,
                (count,),
                device=self.device,
            )
            self._delay_steps[env_ids] = torch.clamp(
                torch.round(center + (sampled - center) * scales),
                self.cfg.min_delay_policy_steps,
                self.cfg.max_delay_policy_steps,
            ).long()
        else:
            self._delay_steps[env_ids] = self.cfg.min_delay_policy_steps


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """差速动作映射、电机响应和随机延迟配置。"""

    class_type: type[ActionTerm] = DifferentialDriveAction
    asset_name: str = MISSING
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    wheel_radius_m: float = 0.032
    wheel_separation_m: float = 0.175
    max_linear_speed_mps: float = 0.10
    max_angular_speed_radps: float = 1.047197551
    enforce_planar_motion: bool = True
    max_linear_accel_mps2: float = 0.6
    max_linear_decel_mps2: float = 0.6
    max_angular_accel_radps2: float = 1.5
    max_angular_decel_radps2: float = 1.5
    motor_time_constant_sec: float = 0.15
    left_motor_gain: float = 1.0
    right_motor_gain: float = 1.0
    motor_time_constant_sec_min: float = 0.15
    motor_time_constant_sec_max: float = 0.15
    left_motor_gain_min: float = 1.0
    left_motor_gain_max: float = 1.0
    right_motor_gain_min: float = 1.0
    right_motor_gain_max: float = 1.0
    max_linear_accel_mps2_min: float = 0.6
    max_linear_accel_mps2_max: float = 0.6
    max_linear_decel_mps2_min: float = 0.6
    max_linear_decel_mps2_max: float = 0.6
    max_angular_accel_radps2_min: float = 1.5
    max_angular_accel_radps2_max: float = 1.5
    max_angular_decel_radps2_min: float = 1.5
    max_angular_decel_radps2_max: float = 1.5
    min_delay_policy_steps: int = 0
    max_delay_policy_steps: int = 2
    randomize_dynamics: bool = True
    randomize_delay: bool = True
