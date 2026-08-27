"""在环境自动 reset 前捕获正式评估的逐回合真值。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import RecorderTerm

from .observations import _yaw_from_quaternion


class NavigationEvaluationRecorder(RecorderTerm):
    """将终止原因与轨迹指标保存到环境内存，供评估入口读取。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._car_completed_episodes = []
        self.max_steps = int(env.max_episode_length)
        self.cross_track = torch.zeros(
            (env.num_envs, self.max_steps), device=env.device
        )
        self.action_delta = torch.zeros_like(self.cross_track)
        self.minimum_clearance = torch.full((env.num_envs,), torch.inf, device=env.device)

    def record_post_step(self):
        indices = torch.clamp(
            self._env.episode_length_buf - 1, 0, self.max_steps - 1
        ).long()
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self.cross_track[env_ids, indices] = self._env._car_cross_track_error
        delta = torch.linalg.vector_norm(
            self._env.action_manager.action - self._env.action_manager.prev_action, dim=-1
        )
        self.action_delta[env_ids, indices] = delta
        clearance = self._env._car_min_clearance
        self.minimum_clearance.copy_(torch.minimum(self.minimum_clearance, clearance))
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = [int(value) for value in env_ids]
        if not env_ids:
            return None, None
        device_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        robot = self._env.scene["robot"]
        command = self._env.command_manager.get_term("navigation")
        yaw = _yaw_from_quaternion(robot.data.root_quat_w)
        position_error = torch.linalg.vector_norm(
            command.command[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
        )
        yaw_error = torch.abs(
            torch.atan2(
                torch.sin(command.command[:, 2] - yaw),
                torch.cos(command.command[:, 2] - yaw),
            )
        )
        termination = self._env.termination_manager
        terms = {
            name: termination.get_term(name)[device_ids].detach().cpu()
            for name in (
                "goal_reached",
                "collision",
                "path_deviation",
                "stalled",
                "invalid_input",
                "time_out",
            )
        }
        lengths = self._env.episode_length_buf[device_ids].detach().cpu().tolist()
        cross_track = self.cross_track[device_ids].detach().cpu()
        action_delta = self.action_delta[device_ids].detach().cpu()
        position_error = position_error[device_ids].detach().cpu()
        yaw_error = yaw_error[device_ids].detach().cpu()
        minimum_clearance = self.minimum_clearance[device_ids].detach().cpu()
        path_lengths = command.metrics["path_length_m"][device_ids].detach().cpu()
        static_collision = getattr(self._env, "_car_static_collision", None)
        dynamic_collision = getattr(self._env, "_car_dynamic_collision", None)
        static_collision = (
            static_collision[device_ids].detach().cpu()
            if static_collision is not None else terms["collision"]
        )
        dynamic_collision = (
            dynamic_collision[device_ids].detach().cpu()
            if dynamic_collision is not None else torch.zeros_like(terms["collision"])
        )
        seeds = command.current_seed[env_ids].tolist()
        for row, env_id in enumerate(env_ids):
            length = min(int(lengths[row]), self.max_steps)
            if length <= 0:
                continue
            active_reasons = [
                name for name, value in terms.items() if bool(value[row])
            ]
            self._env._car_completed_episodes.append(
                {
                    "success": bool(terms["goal_reached"][row]),
                    "collision": bool(terms["collision"][row]),
                    "static_collision": bool(static_collision[row]),
                    "dynamic_collision": bool(dynamic_collision[row]),
                    "timeout": bool(terms["time_out"][row]),
                    "stall": bool(terms["stalled"][row]),
                    "path_deviation": bool(terms["path_deviation"][row]),
                    "invalid_input": bool(terms["invalid_input"][row]),
                    "termination_reason": "+".join(active_reasons) or "unknown",
                    "path_length_m": float(path_lengths[row]),
                    "cross_track_errors_m": cross_track[row, :length].tolist(),
                    "goal_position_error_m": float(position_error[row]),
                    "goal_yaw_error_rad": float(yaw_error[row]),
                    "episode_time_sec": float(length * self._env.step_dt),
                    "action_deltas": action_delta[row, :length].tolist(),
                    "minimum_clearance_m": float(minimum_clearance[row]),
                    "seed": int(seeds[row]),
                }
            )
        self.cross_track[device_ids] = 0.0
        self.action_delta[device_ids] = 0.0
        self.minimum_clearance[device_ids] = torch.inf
        return None, None
