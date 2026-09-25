import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def exponential_kernel(
    data_ptr,
    n_elements,
    lambd,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill tensor with exponential distribution: x = -log(1-u)/lambd"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    u = tl.rand(seed, offsets)
    x = -tl.log(1.0 - u) / lambd

    tl.store(data_ptr + offsets, x, mask=mask)


def exponential_(self: torch.Tensor, lambd: float = 1.0, *, generator=None) -> torch.Tensor:
    """Implement aten::exponential_"""
    n_elements = self.numel()
    if n_elements == 0:
        return self

    if generator is not None:
        seed = generator.seed()
    else:
        seed = self.data_ptr() % (2**31 - 1)

    lambd = float(lambd)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exponential_kernel[grid](self, n_elements, lambd, seed)

    return self