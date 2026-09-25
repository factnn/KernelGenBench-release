"""
vLLM marlin_int4_fp8_preprocess baseline wrapper.
"""
import torch
from vllm import _custom_ops


def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False
) -> None:
    """Wrapper for vLLM marlin_int4_fp8_preprocess implementation."""
    return _custom_ops.marlin_int4_fp8_preprocess(
        qweight,
        qzeros_or_none,
        inplace
    )
