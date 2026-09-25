"""
vLLM topk_softmax baseline wrapper.
"""
import torch
from vllm import _custom_ops


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Wrapper for vLLM topk_softmax implementation."""
    _custom_ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
    )
