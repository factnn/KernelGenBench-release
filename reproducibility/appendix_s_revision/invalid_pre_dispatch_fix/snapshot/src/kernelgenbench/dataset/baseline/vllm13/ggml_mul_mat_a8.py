"""
vLLM ggml_mul_mat_a8 baseline wrapper.
"""
import torch
from vllm import _custom_ops


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int
) -> torch.Tensor:
    """Wrapper for vLLM ggml_mul_mat_a8 implementation."""
    return _custom_ops.ggml_mul_mat_a8(
        W,
        X,
        quant_type,
        row
    )
