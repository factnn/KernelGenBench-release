import torch
import triton
import triton.language as tl


@triton.jit
def _merge_attn_states_kernel(
    output_ptr,
    output_lse_ptr,
    prefix_output_ptr,
    prefix_lse_ptr,
    suffix_output_ptr,
    suffix_lse_ptr,
    num_tokens,
    num_heads,
    prefix_head_stride,
    output_head_stride,
    head_size,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    """Merge prefix and suffix attention outputs using log-sum-exp weighting."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    p_lse = tl.load(prefix_lse_ptr + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse_ptr + head_idx * num_tokens + token_idx)

    # Handle inf -> -inf for FA2 compatibility
    p_lse = tl.where(p_lse == float("inf"), float("-inf"), p_lse)
    s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)

    max_lse = tl.maximum(p_lse, s_lse)

    p_lse_sub = p_lse - max_lse
    s_lse_sub = s_lse - max_lse
    p_se = tl.exp(p_lse_sub)
    s_se = tl.exp(s_lse_sub)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse_ptr + head_idx * num_tokens + token_idx, out_lse)

    p_scale = p_se / out_se
    s_scale = s_se / out_se

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < head_size

    prefix_base = (
        prefix_output_ptr
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
    )
    suffix_base = (
        suffix_output_ptr
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
    )
    out_base = (
        output_ptr
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
    )

    p_out = tl.load(prefix_base + head_arange, mask=head_mask, other=0.0)
    s_out = tl.load(suffix_base + head_arange, mask=head_mask, other=0.0)

    out = p_out * p_scale + s_out * s_scale
    tl.store(out_base + head_arange, out.to(p_out.dtype), mask=head_mask)


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None
) -> None:
    num_tokens = output.shape[0]
    num_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    _merge_attn_states_kernel[(num_tokens, num_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        num_tokens,
        num_heads,
        prefix_head_stride,
        output_head_stride,
        head_size,
        PADDED_HEAD_SIZE=padded_head_size,
        OUTPUT_LSE=output_lse is not None,
    )