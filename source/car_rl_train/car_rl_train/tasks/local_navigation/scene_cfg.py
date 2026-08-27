"""包含车辆、本地地面和边界墙的 Isaac 仿真场景。"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CAR_USD = PROJECT_ROOT / "assets" / "usd" / "car.usd"


def _wall(prim_path: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
    """创建带碰撞和可视材质的静态边界墙配置。"""

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class CarNavigationSceneCfg(InteractiveSceneCfg):
    """局部导航并行场景中的车辆、离线地面、边界墙和光照。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CAR_USD),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # articulation 根是 base_footprint，不是 base_link；接近地面生成，
            # 防止初始化后自由跌落造成左右轮接触不对称。
            pos=(0.0, 0.0, 0.002),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                # 0.8 kg 小车维持实测 0.344 m/s² 只需约 0.0044 N·m/轮。
                # 旧值 20 N·m 会在接触微小不对称时把底盘掀向一侧。
                effort_limit_sim=0.02,
                velocity_limit_sim=30.0,
                stiffness=0.0,
                damping=1.0,
            )
        },
    )
    wall_north = _wall("{ENV_REGEX_NS}/WallNorth", (6.0, 0.10, 0.40), (0.0, 3.0, 0.20))
    wall_south = _wall("{ENV_REGEX_NS}/WallSouth", (6.0, 0.10, 0.40), (0.0, -3.0, 0.20))
    wall_east = _wall("{ENV_REGEX_NS}/WallEast", (0.10, 6.0, 0.40), (3.0, 0.0, 0.20))
    wall_west = _wall("{ENV_REGEX_NS}/WallWest", (0.10, 6.0, 0.40), (-3.0, 0.0, 0.20))
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )
