"""确定性控制器 ONNX 导出及数值一致性验证。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contract.schema import INPUT_NAME, INPUT_SHAPE, ONNX_OPSET, OUTPUT_NAME, OUTPUT_SHAPE


class ExportedController(torch.nn.Module):
    """仅保留确定性 actor，并对最终动作再次执行安全裁剪。"""

    def __init__(self, actor: torch.nn.Module):
        """保存待导出的 actor 子模块。"""

        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """将单条 86 维观测映射为范围在 [-1,1] 的两维动作。"""

        return torch.clamp(self.actor(observation), -1.0, 1.0)


def export_controller(actor: torch.nn.Module, output_path: str | Path) -> Path:
    """以 opset 18 和固定 shape 导出控制器 ONNX。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportedController(actor).eval().cpu()
    with torch.inference_mode():
        probe = wrapper(torch.zeros(*INPUT_SHAPE, dtype=torch.float32))
    if list(probe.shape) != OUTPUT_SHAPE or not torch.isfinite(probe).all():
        raise ValueError("actor does not satisfy controller output contract")
    torch.onnx.export(
        wrapper,
        torch.zeros(*INPUT_SHAPE, dtype=torch.float32),
        str(output),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=ONNX_OPSET,
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )
    return output


def inspect_onnx_contract(path: str | Path) -> dict[str, Any]:
    """检查 ONNX 合法性、opset、张量名称及静态输入输出维度。"""

    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("", 0) != ONNX_OPSET:
        raise ValueError(f"expected ONNX opset {ONNX_OPSET}, got {opsets.get('')}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("controller ONNX must have exactly one input and one output")

    def tensor_spec(value) -> tuple[str, list[int]]:
        """提取张量名称与静态维度，并拒绝动态维度。"""

        dimensions: list[int] = []
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise ValueError("dynamic or non-positive dimensions are forbidden")
            dimensions.append(int(dimension.dim_value))
        return value.name, dimensions

    input_spec = tensor_spec(model.graph.input[0])
    output_spec = tensor_spec(model.graph.output[0])
    if input_spec != (INPUT_NAME, INPUT_SHAPE) or output_spec != (OUTPUT_NAME, OUTPUT_SHAPE):
        raise ValueError(f"invalid ONNX tensor contract: {input_spec} -> {output_spec}")
    return {"opset": ONNX_OPSET, "input": input_spec, "output": output_spec}


def validate_onnx_numerics(
    actor: torch.nn.Module,
    onnx_path: str | Path,
    observations: np.ndarray,
    warmup: int = 20,
) -> dict[str, float]:
    """逐条比较 PyTorch 与 ONNX Runtime 输出并统计 p99 延迟。"""

    import onnxruntime as ort

    inspect_onnx_contract(onnx_path)
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 86 or not np.isfinite(values).all():
        raise ValueError("observations must be finite float32 [N,86]")
    wrapper = ExportedController(actor).eval().cpu()
    with torch.inference_mode():
        expected = wrapper(torch.from_numpy(values)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # 模型契约固定为静态 [1,86]，因此必须逐条执行观测，不能改成动态 batch。
    actual_rows: list[np.ndarray] = []
    latencies_ms: list[float] = []
    for index, row in enumerate(values):
        batch = row.reshape(1, 86)
        started = time.perf_counter()
        output = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup:
            latencies_ms.append(elapsed)
        actual_rows.append(output[0])
    actual = np.asarray(actual_rows, dtype=np.float32)
    difference = np.abs(expected - actual)
    if not np.isfinite(actual).all() or np.any(actual < -1.0) or np.any(actual > 1.0):
        raise ValueError("ONNX output is non-finite or outside [-1,1]")
    return {
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "p99_inference_ms": float(np.percentile(latencies_ms or [0.0], 99)),
    }
