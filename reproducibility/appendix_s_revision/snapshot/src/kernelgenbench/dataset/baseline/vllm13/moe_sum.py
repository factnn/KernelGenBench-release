"""
vLLM moe_sum baseline wrapper.
"""
import torch
from vllm import _custom_ops


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor
) -> None:
    """Wrapper for vLLM moe_sum implementation."""
    _custom_ops.moe_sum(
        input,
        output
    )
