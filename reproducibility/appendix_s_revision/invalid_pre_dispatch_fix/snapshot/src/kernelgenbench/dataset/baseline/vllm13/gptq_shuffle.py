"""
vLLM gptq_shuffle baseline wrapper.
"""
import torch
from vllm import _custom_ops


def gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int
) -> None:
    """Wrapper for vLLM gptq_shuffle implementation."""
    _custom_ops.gptq_shuffle(
        q_weight,
        q_perm,
        bit
    )
