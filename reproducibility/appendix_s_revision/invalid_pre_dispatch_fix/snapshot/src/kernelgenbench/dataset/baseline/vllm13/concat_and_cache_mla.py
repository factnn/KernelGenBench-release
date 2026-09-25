"""
vLLM concat_and_cache_mla baseline wrapper.
"""
import torch
from vllm import _custom_ops


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor
) -> None:
    """Wrapper for vLLM concat_and_cache_mla implementation."""
    _custom_ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        scale
    )
