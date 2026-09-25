"""vLLM copy_blocks baseline wrapper."""
import torch
from vllm import _custom_ops


def copy_blocks(
    key_caches: list,
    value_caches: list,
    block_mapping: torch.Tensor
) -> None:
    _custom_ops.copy_blocks(key_caches, value_caches, block_mapping)
