import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def cos_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.cos(x.to(tl.float32))
    tl.store(output_ptr + offsets, output, mask=mask)


def cos(self: torch.Tensor) -> torch.Tensor:
    """实现 aten::cos"""
    self = self.contiguous()
    output = torch.empty_like(self)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_kernel[grid](self, output, n_elements)
    return output


def cos_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """实现 aten::cos.out"""
    x = self.contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_kernel[grid](x, out, n_elements)
    return out


def cos_int(a: int) -> float:
    """实现 aten::cos.int"""
    return math.cos(float(a))


def cos_float(a: float) -> float:
    """实现 aten::cos.float"""
    return math.cos(a)


def cos_complex(a: complex) -> complex:
    """实现 aten::cos.complex"""
    import cmath
    return cmath.cos(a)


def cos_Scalar(a) -> float:
    """实现 aten::cos.Scalar"""
    return math.cos(float(a))