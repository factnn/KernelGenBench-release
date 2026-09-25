"""
vLLM grouped_topk baseline wrapper.
"""
import torch
from vllm import _custom_ops


def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0
) -> None:
    """Perform grouped top-k routing for mixture of experts."""
    return _custom_ops.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func
    )
