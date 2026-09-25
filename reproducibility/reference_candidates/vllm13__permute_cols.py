import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def _permute_cols_kernel(
    output_ptr, input_ptr, perm_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for permute_cols: output[:, j] = input[:, perm[j]].

    Loads perm indices, gathers from input, stores to output.
    Stores are always coalesced. Loads are coalesced for sequential perm
    (identity, reverse) and scattered for random perm.
    """
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    perm_idx = tl.load(perm_ptr + col_offsets, mask=mask, other=0)
    input_ptrs = input_ptr + row_idx * n_cols + perm_idx
    vals = tl.load(input_ptrs, mask=mask, other=0.0)

    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, vals, mask=mask)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Triton implementation of vLLM permute_cols.

    Reorders columns of `a` according to `perm`:
        output[:, j] = a[:, perm[j]]

    Args:
        a: Input tensor of shape (rows, cols).
        perm: Permutation indices tensor of shape (cols,), dtype int32.

    Returns:
        Output tensor of same shape and dtype as `a`.
    """
    n_rows, n_cols = a.shape
    out = torch.empty_like(a)

    if perm.device != a.device:
        perm = perm.to(a.device)

    grid = lambda meta: (n_rows, triton.cdiv(n_cols, meta['BLOCK_SIZE']))

    _permute_cols_kernel[grid](
        out, a, perm,
        n_rows, n_cols,
    )

    return out