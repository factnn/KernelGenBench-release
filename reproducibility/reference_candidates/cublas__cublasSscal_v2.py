import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def cublasSscal_v2_kernel(n, alpha, x_ptr, incx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx

    x = tl.load(x_ptr + x_idx, mask=mask)
    result = alpha * x
    tl.store(x_ptr + x_idx, result, mask=mask)


def cublasSscal_v2(n, alpha, x, incx):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    cublasSscal_v2_kernel[grid](n, alpha, x, incx)
    return x