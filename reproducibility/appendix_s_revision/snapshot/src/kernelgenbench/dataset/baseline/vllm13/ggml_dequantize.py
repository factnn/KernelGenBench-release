"""
vLLM ggml_dequantize baseline wrapper.
"""
import torch
from vllm import _custom_ops


def ggml_dequantize(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype
) -> torch.Tensor:
    """Wrapper for vLLM ggml_dequantize implementation."""
    return _custom_ops.ggml_dequantize(
        W, quant_type, m, n, dtype
    )
