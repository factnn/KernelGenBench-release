"""
vLLM shuffle_rows baseline wrapper.
"""
import torch
from vllm import _custom_ops


def shuffle_rows(
    input_tensor: torch.Tensor,
    dst2src_map: torch.Tensor
) -> None:
    """Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor."""
    return _custom_ops.shuffle_rows(
        input_tensor,
        dst2src_map
    )
