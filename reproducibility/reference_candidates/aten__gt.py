import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gt_kernel(
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
    output = (x.to(tl.float32) > y.to(tl.float32)).to(tl.int8)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gt_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = (x.to(tl.float32) > scalar_val).to(tl.int8)
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# Tensor overloads
# ============================================================================

def gt_Tensor(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """aten::gt.Tensor(Tensor self, Tensor other) -> Tensor"""
    self, other = torch.broadcast_tensors(self, other)
    output = torch.empty(self.shape, dtype=torch.bool, device=self.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gt_kernel[grid](self, other, output, n_elements)
    return output


def gt_Scalar(self: torch.Tensor, other) -> torch.Tensor:
    """aten::gt.Scalar(Tensor self, Scalar other) -> Tensor"""
    output = torch.empty(self.shape, dtype=torch.bool, device=self.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    scalar_val = float(other)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gt_scalar_kernel[grid](self, output, n_elements, scalar_val)
    return output


def gt_Scalar_out(self: torch.Tensor, other, *, out: torch.Tensor) -> torch.Tensor:
    """aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = gt_Scalar(self, other)
    out.copy_(result)
    return out


def gt_Tensor_out(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = gt_Tensor(self, other)
    out.copy_(result)
    return out


# ============================================================================
# Non-tensor overloads
# ============================================================================

def gt_int(a: int, b: int) -> bool:
    return a > b


def gt_float(a: float, b: float) -> bool:
    return a > b


def gt_int_float(a: int, b: float) -> bool:
    return float(a) > b


def gt_float_int(a: float, b: int) -> bool:
    return a > float(b)


def gt(a, b) -> bool:
    return a > b


def gt_str(a: str, b: str) -> bool:
    return a > b