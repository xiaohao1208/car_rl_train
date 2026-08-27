"""控制器专用局部导航 ManagerBasedRLEnv 配置。"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    RecorderTermCfg,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.utils import configclass

from . import curriculum, observations, physics_events, rewards, terminations
from .actions import DifferentialDriveActionCfg
from .commands import NavigationCommandCfg
from .evaluation_recorder import NavigationEvaluationRecorder
from .scene_cfg import CarNavigationSceneCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg


@configclass
class ActionsCfg:
    """环境动作管理器配置。"""

    drive = DifferentialDriveActionCfg(asset_name="robot")


@configclass
class CommandsCfg:
    """导航任务指令管理器配置。"""

    navigation = NavigationCommandCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """控制器观测组配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """仅向策略暴露一个严格 86 维的拼接观测项。"""

        controller = ObsTerm(func=observations.controller_observation)

        def __post_init__(self) -> None:
            """关闭 Isaac 自动扰动，统一由契约观测函数控制噪声。"""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """路径跟随、安全、动作平滑和终止事件奖励配置。"""

    progress = RewTerm(func=rewards.path_progress, weight=4.0)
    cross_track = RewTerm(func=rewards.cross_track_error, weight=-0.8)
    heading = RewTerm(func=rewards.heading_error_cost, weight=-0.5)
    obstacle = RewTerm(func=rewards.obstacle_proximity_cost, weight=-0.60)
    near_goal_yaw = RewTerm(func=rewards.near_goal_yaw_progress, weight=1.0)
    action_delta = RewTerm(func=rewards.action_delta_squared, weight=-0.03)
    angular_action = RewTerm(func=rewards.angular_action_squared, weight=-0.01)
    stopped = RewTerm(func=rewards.stopped_away_from_goal, weight=-0.02)
    goal = RewTerm(
        func=rewards.terminal_event,
        weight=16.0,
        params={"term_name": "goal_reached"},
    )
    collision = RewTerm(
        func=rewards.terminal_event,
        weight=-24.0,
        params={"term_name": "collision"},
    )
    path_deviation = RewTerm(
        func=rewards.terminal_event,
        weight=-8.0,
        params={"term_name": "path_deviation"},
    )
    stall = RewTerm(
        func=rewards.terminal_event,
        weight=-16.0,
        params={"term_name": "stalled"},
    )
    timeout = RewTerm(
        func=rewards.terminal_event,
        weight=-1.0,
        params={"term_name": "time_out"},
    )
    invalid = RewTerm(
        func=rewards.terminal_event,
        weight=-12.0,
        params={"term_name": "invalid_input"},
    )


@configclass
class TerminationsCfg:
    """超时、到达、碰撞、偏航、停滞及输入异常终止配置。"""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=terminations.goal_reached)
    collision = DoneTerm(func=terminations.collision)
    path_deviation = DoneTerm(func=terminations.path_deviation)
    stalled = DoneTerm(func=terminations.stalled)
    invalid_input = DoneTerm(func=terminations.invalid_input)


@configclass
class CurriculumCfg:
    """通用多阶段课程学习管理器配置。"""

    stage = CurrTerm(
        func=curriculum.curriculum_stage,
        params={
            "stages": [
                {
                    "stage": index,
                    "success_rate": 0.90,
                    "max_collision_rate": 0.03,
                    "min_stage_episodes": 500,
                }
                for index in range(8)
            ],
            "window_episodes": 500,
            "consecutive_passes": 1,
        },
    )


@configclass
class EventsCfg:
    """场景复位以及实车物理参数启动事件。"""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    total_mass = EventTerm(
        func=physics_events.randomize_total_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "total_mass_range_kg": (1.0, 4.0),
        },
    )
    wheel_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_wheel_link", "right_wheel_link"],
            ),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    caster_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["front_caster_link", "back_caster_link"],
            ),
            "static_friction_range": (0.05, 0.05),
            "dynamic_friction_range": (0.05, 0.05),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class CarLocalNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """训练环境的场景、管理器、频率和回合上限总配置。"""

    scene: CarNavigationSceneCfg = CarNavigationSceneCfg(num_envs=256, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        """设置 120 Hz 物理、15 Hz 策略和 30 秒回合限制。"""

        self.decimation = 8
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


class CarLocalNavigationPlayEnvCfg(CarLocalNavigationEnvCfg):
    """关闭训练扰动、减少并行数量的确定性演示环境。"""

    def __post_init__(self) -> None:
        """继承训练契约，同时切换到最终课程阶段和 Play 参数。"""

        super().__post_init__()
        self.scene.num_envs = 16
        self.actions.drive.randomize_dynamics = False
        self.actions.drive.randomize_delay = False
        self.commands.navigation.curriculum_stage = 7
        self.commands.navigation.randomize_observations = False
        self.events.total_mass.params["total_mass_range_kg"] = (2.5, 2.5)
        self.events.wheel_material.params["static_friction_range"] = (0.85, 0.85)
        self.events.wheel_material.params["dynamic_friction_range"] = (0.70, 0.70)


@configclass
class EvaluationRecordersCfg(RecorderManagerBaseCfg):
    """只在内存捕获评估轨迹，不生成 Isaac HDF5 数据集。"""

    navigation = RecorderTermCfg(class_type=NavigationEvaluationRecorder)
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    export_in_close = False


@configclass
class CarLocalNavigationEvaluationEnvCfg(CarLocalNavigationPlayEnvCfg):
    """固定测试种子并在自动 reset 前记录逐回合指标。"""

    recorders: EvaluationRecordersCfg = EvaluationRecordersCfg()
