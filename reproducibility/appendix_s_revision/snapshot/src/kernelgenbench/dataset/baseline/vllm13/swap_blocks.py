"""
vLLM swap_blocks baseline wrapper.
"""
import torch
from vllm import _custom_ops


def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor
) -> None:
    """Copy specific blocks from one tensor to another."""
    _custom_ops.swap_blocks(
        src,
        dst,
        block_mapping
    )
