import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def le_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x <= y

    tl.store(output_ptr + offsets, output, mask=mask)


def _le_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Core implementation for element-wise less-than-or-equal."""
    self, other = torch.broadcast_tensors(self, other)
    self = self.contiguous()
    other = other.contiguous()

    output = torch.empty_like(self, dtype=torch.bool)
    n_elements = output.numel()

    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    le_kernel[grid](self, other, output, n_elements)

    return output


def le_Tensor(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Implement aten::le.Tensor"""
    return _le_impl(self, other)


def le_Scalar(self: torch.Tensor, other) -> torch.Tensor:
    """Implement aten::le.Scalar"""
    other_tensor = torch.tensor(other, device=self.device, dtype=self.dtype)
    return _le_impl(self, other_tensor)


def le_Scalar_out(self: torch.Tensor, other, *, out: torch.Tensor) -> torch.Tensor:
    """Implement aten::le.Scalar_out"""
    result = le_Scalar(self, other)
    out.copy_(result)
    return out


def le_Tensor_out(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """Implement aten::le.Tensor_out"""
    result = le_Tensor(self, other)
    out.copy_(result)
    return out


def le_int(a: int, b: int) -> bool:
    """Implement aten::le.int"""
    return a <= b


def le_float(a: float, b: float) -> bool:
    """Implement aten::le.float"""
    return a <= b


def le_int_float(a: int, b: float) -> bool:
    """Implement aten::le.int_float"""
    return a <= b


def le_float_int(a: float, b: int) -> bool:
    """Implement aten::le.float_int"""
    return a <= b


def le(a, b) -> bool:
    """Implement aten::le (Scalar, Scalar)"""
    return a <= b


def le_str(a: str, b: str) -> bool:
    """Implement aten::le.str"""
    return a <= b