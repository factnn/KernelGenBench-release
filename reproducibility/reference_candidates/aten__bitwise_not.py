import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def bitwise_not_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = ~x
    tl.store(output_ptr + offsets, output, mask=mask)


def bitwise_not(self: torch.Tensor) -> torch.Tensor:
    """实现 aten::bitwise_not"""
    output = torch.empty_like(self)
    n_elements = output.numel()

    if n_elements == 0:
        return output

    input_contiguous = self.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bitwise_not_kernel[grid](input_contiguous, output, n_elements)

    return output


def bitwise_not_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """实现 aten::bitwise_not.out"""
    result = bitwise_not(self)
    out.copy_(result)
    return out