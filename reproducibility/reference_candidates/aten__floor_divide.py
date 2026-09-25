import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
    reset_to_zero=['output_ptr'],
)
@triton.jit
def _floor_divide_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    is_float: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    if is_float:
        output = tl.math.floor(x / y)
    else:
        is_zero = (y == 0)
        safe_y = tl.where(is_zero, 1, y)
        q = x // safe_y
        r = x % safe_y
        correction = (r != 0) & ((x > 0) != (y > 0))
        normal_result = q - correction.to(tl.int32)
        zero_result = tl.where(x >= 0, -1, -2)
        output = tl.where(is_zero, zero_result, normal_result)

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
    reset_to_zero=['output_ptr'],
)
@triton.jit
def _floor_divide_kernel_scalar(
    x_ptr,
    y,
    output_ptr,
    n_elements,
    is_float: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if is_float:
        output = tl.math.floor(x / y)
    else:
        is_zero = (y == 0)
        safe_y = tl.where(is_zero, 1, y)
        q = x // safe_y
        r = x % safe_y
        correction = (r != 0) & ((x > 0) != (y > 0))
        normal_result = q - correction.to(tl.int32)
        zero_result = tl.where(x >= 0, -1, -2)
        output = tl.where(is_zero, zero_result, normal_result)

    tl.store(output_ptr + offsets, output, mask=mask)


def floor_divide(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """aten::floor_divide(Tensor self, Tensor other) -> Tensor"""
    self, other = torch.broadcast_tensors(self, other)
    if not self.is_contiguous():
        self = self.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()

    output = torch.empty_like(self)
    n_elements = output.numel()
    is_float = self.dtype.is_floating_point

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _floor_divide_kernel[grid](self, other, output, n_elements, is_float=is_float)
    return output


def floor_divide_Scalar(self: torch.Tensor, other: float) -> torch.Tensor:
    """aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor"""
    if not self.is_contiguous():
        self = self.contiguous()

    output = torch.empty_like(self)
    n_elements = output.numel()
    is_float = self.dtype.is_floating_point

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _floor_divide_kernel_scalar[grid](self, other, output, n_elements, is_float=is_float)
    return output


def floor_divide_out(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = floor_divide(self, other)
    out.copy_(result)
    return out


def floor_divide_Scalar_out(self: torch.Tensor, other: float, *, out: torch.Tensor) -> torch.Tensor:
    """aten::floor_divide.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = floor_divide_Scalar(self, other)
    out.copy_(result)
    return out