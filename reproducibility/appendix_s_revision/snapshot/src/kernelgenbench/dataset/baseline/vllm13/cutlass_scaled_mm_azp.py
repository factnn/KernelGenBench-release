"""
vLLM cutlass_scaled_mm_azp baseline wrapper.
"""
import torch
from vllm import _custom_ops


def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    """:param azp_adj: In the per-tensor case, this should include the azp."""
    return _custom_ops.cutlass_scaled_mm_azp(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype,
        azp_adj,
        azp,
        bias
    )
