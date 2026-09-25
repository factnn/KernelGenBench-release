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
def eq_kernel(
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
    output = (x == y).to(tl.int8)
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
def eq_scalar_kernel(
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
    output = (x == scalar_val).to(tl.int8)
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# Tensor overloads
# ============================================================================

def eq_Tensor(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """aten::eq.Tensor(Tensor self, Tensor other) -> Tensor"""
    self, other = torch.broadcast_tensors(self, other)
    output = torch.empty(self.shape, dtype=torch.bool, device=self.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    eq_kernel[grid](self, other, output, n_elements)
    return output


def eq_Scalar(self: torch.Tensor, other) -> torch.Tensor:
    """aten::eq.Scalar(Tensor self, Scalar other) -> Tensor"""
    output = torch.empty(self.shape, dtype=torch.bool, device=self.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    if isinstance(other, bool):
        scalar_val = float(other)
    elif isinstance(other, complex):
        other_tensor = torch.full(self.shape, other, dtype=self.dtype, device=self.device)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        eq_kernel[grid](self, other_tensor, output, n_elements)
        return output
    else:
        scalar_val = float(other)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    eq_scalar_kernel[grid](self, output, n_elements, scalar_val)
    return output


def eq_Scalar_out(self: torch.Tensor, other, *, out: torch.Tensor) -> torch.Tensor:
    """aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = eq_Scalar(self, other)
    out.copy_(result)
    return out


def eq_Tensor_out(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"""
    result = eq_Tensor(self, other)
    out.copy_(result)
    return out


# ============================================================================
# Non-tensor overloads
# ============================================================================

def eq_int_list(a: list, b: list) -> bool:
    return a == b

def eq_device(a, b) -> bool:
    return str(a) == str(b)

def eq_bool(a: bool, b: bool) -> bool:
    return a == b

def eq_enum(a, b) -> bool:
    return a == b

def eq_int(a: int, b: int) -> bool:
    return a == b

def eq_complex(a: complex, b: complex) -> bool:
    return a == b

def eq_float(a: float, b: float) -> bool:
    return a == b

def eq_int_float(a: int, b: float) -> bool:
    return float(a) == b

def eq_float_int(a: float, b: int) -> bool:
    return a == float(b)

def eq_float_complex(a: float, b: complex) -> bool:
    return complex(a) == b

def eq_complex_float(a: complex, b: float) -> bool:
    return a == complex(b)

def eq(a, b) -> bool:
    return a == b

def eq_str(a: str, b: str) -> bool:
    return a == b

def eq_float_list(a: list, b: list) -> bool:
    return a == b

def eq_Tensor_list(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ta, tb in zip(a, b):
        if not torch.equal(ta, tb):
            return False
    return True

def eq_bool_list(a: list, b: list) -> bool:
    return a == b

def eq_str_list(a: list, b: list) -> bool:
    return a == b