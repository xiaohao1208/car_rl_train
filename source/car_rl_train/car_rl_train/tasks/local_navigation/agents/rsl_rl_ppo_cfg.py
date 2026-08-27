"""与 rl_plan.md 保持一致的 PPO 配置。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .. import symmetry


@configclass
class CarPolicyCfg(RslRlPpoActorCriticCfg):
    """项目唯一的 split-fusion actor-critic 配置。"""

    class_name: str = "CarActorCritic"
    min_noise_std: float = 0.04
    max_noise_std: float = 0.10


@configclass
class CarPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """项目 PPO 的单调学习率衰减参数。"""

    min_learning_rate: float = 1.0e-5
    lr_decay_half_life: float = 50.0
    symmetry_contract: str = "car_mirror_v1"


@configclass
class CarNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """局部导航 PPO 的 rollout、优化和 checkpoint 配置。"""

    class_name = "CarOnPolicyRunner"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 8000
    save_interval = 50
    experiment_name = "car_local_navigation"
    seed = 42
    clip_actions = 1.0
    policy = CarPolicyCfg(
        init_noise_std=0.08,
        noise_std_type="bounded_logit",
        min_noise_std=0.04,
        max_noise_std=0.10,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = CarPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0002,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        min_learning_rate=1.0e-5,
        lr_decay_half_life=50.0,
        symmetry_contract="car_mirror_v1",
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=symmetry.mirror_navigation_batch,
        ),
    )
