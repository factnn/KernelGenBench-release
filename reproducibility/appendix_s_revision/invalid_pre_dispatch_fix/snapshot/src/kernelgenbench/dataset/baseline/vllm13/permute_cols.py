"""
vLLM permute_cols baseline wrapper.
"""
import torch
from vllm import _custom_ops


def permute_cols(
    a: torch.Tensor,
    perm: torch.Tensor
) -> torch.Tensor:
    """Wrapper for vLLM permute_cols implementation."""
    return _custom_ops.permute_cols(
        a,
        perm
    )
