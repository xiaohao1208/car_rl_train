"""car_rl_train 包入口；无 Isaac/Gym 时仍允许使用离线标定工具。"""

try:
    from .tasks import register_tasks
except ModuleNotFoundError as error:
    if error.name != "gymnasium":
        raise
    _missing_gym_error = error

    def register_tasks() -> None:
        """在精简 Python 环境中给出明确依赖错误。"""

        raise ModuleNotFoundError(
            "训练环境注册需要 gymnasium/Isaac Lab；离线标定导入不需要"
        ) from _missing_gym_error
else:
    register_tasks()

__all__ = ["register_tasks"]
