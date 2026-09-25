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
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def full_kernel(
    output_ptr,
    fill_value_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    fill_value = tl.load(fill_value_ptr)
    tl.store(output_ptr + offsets, fill_value, mask=mask)


def full_names(size, fill_value, *, names=None, dtype=None, layout=None, device=None, pin_memory=None):
    """实现 aten::full.names"""
    return full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    """实现 aten::full"""
    # Determine output dtype
    if dtype is not None:
        output_dtype = dtype
    elif isinstance(fill_value, torch.Tensor):
        output_dtype = fill_value.dtype
    elif isinstance(fill_value, bool):
        output_dtype = torch.bool
    elif isinstance(fill_value, int):
        output_dtype = torch.int64
    elif isinstance(fill_value, float):
        output_dtype = torch.float32
    elif isinstance(fill_value, complex):
        output_dtype = torch.complex64
    else:
        output_dtype = torch.float32

    # Create output tensor
    if device is not None:
        output = torch.empty(size, dtype=output_dtype, layout=layout if layout is not None else torch.strided, device=device, pin_memory=pin_memory if pin_memory is not None else False)
    else:
        output = torch.empty(size, dtype=output_dtype, layout=layout if layout is not None else torch.strided, pin_memory=pin_memory if pin_memory is not None else False)

    n_elements = output.numel()

    if n_elements == 0:
        return output

    # Convert fill_value to a 1-element tensor on the same device as output
    # This preserves exact bit representation (critical for float precision)
    if not isinstance(fill_value, torch.Tensor):
        fill_value_tensor = torch.tensor(fill_value, dtype=output_dtype, device=output.device)
    else:
        fill_value_tensor = fill_value.to(dtype=output_dtype, device=output.device)
    fill_value_tensor = fill_value_tensor.reshape(1)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    full_kernel[grid](output, fill_value_tensor, n_elements)

    return output


def full_names_out(size, fill_value, *, names=None, out=None):
    """实现 aten::full.names_out"""
    result = full_names(size, fill_value, names=names, dtype=out.dtype, layout=out.layout, device=out.device)
    out.copy_(result)
    return out


def full_out(size, fill_value, *, out=None):
    """实现 aten::full.out"""
    result = full(size, fill_value, dtype=out.dtype, layout=out.layout, device=out.device)
    out.copy_(result)
    return out