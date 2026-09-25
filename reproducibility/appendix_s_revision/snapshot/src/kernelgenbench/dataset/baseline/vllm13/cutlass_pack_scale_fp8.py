"""
vLLM cutlass_pack_scale_fp8 baseline wrapper.
"""
import torch
from vllm import _custom_ops


def cutlass_pack_scale_fp8(
    scales: torch.Tensor
) -> torch.Tensor:
    """Wrapper for vLLM cutlass_pack_scale_fp8 implementation."""
    return _custom_ops.cutlass_pack_scale_fp8(
        scales
    )
