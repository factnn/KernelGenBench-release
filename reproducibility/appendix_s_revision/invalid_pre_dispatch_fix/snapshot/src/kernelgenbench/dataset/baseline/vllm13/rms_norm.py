"""
vLLM rms_norm baseline wrapper.
"""
import torch
from vllm import _custom_ops


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None:
    """Wrapper for vLLM rms_norm implementation."""
    _custom_ops.rms_norm(
        out,
        input,
        weight,
        epsilon
    )
