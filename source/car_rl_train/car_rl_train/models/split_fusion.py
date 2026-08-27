"""分别编码雷达与导航状态，再进行特征融合的策略网络。"""

from __future__ import annotations

import torch


def activation_factory(name: str) -> type[torch.nn.Module]:
    """把当前策略支持的激活函数名称转换为 PyTorch 模块。"""

    values = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "silu": torch.nn.SiLU,
    }
    try:
        return values[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


class SplitFusionEncoder(torch.nn.Module):
    """分别提取雷达与导航状态特征后进行融合。"""

    def __init__(
        self,
        observation_dim: int = 86,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
    ):
        """创建雷达分支、导航分支及融合层。"""

        del hidden_dims
        super().__init__()
        if observation_dim != 86:
            raise ValueError("controller observation must have 86 elements")
        act = activation_factory(activation)
        self.lidar = torch.nn.Sequential(
            torch.nn.Linear(72, 128),
            act(),
            torch.nn.Linear(128, 64),
            act(),
        )
        self.navigation = torch.nn.Sequential(
            torch.nn.Linear(14, 64),
            act(),
            torch.nn.Linear(64, 64),
            act(),
        )
        self.fusion = torch.nn.Sequential(torch.nn.Linear(128, 128), act())
        self.output_dim = 128

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """切分 72 维雷达和 14 维状态，并返回融合特征。"""

        if not torch.jit.is_tracing() and observation.shape[-1] != 86:
            raise ValueError(f"expected 86 observations, got {observation.shape}")
        return self.fusion(
            torch.cat((self.lidar(observation[..., :72]), self.navigation(observation[..., 72:])), dim=-1)
        )
