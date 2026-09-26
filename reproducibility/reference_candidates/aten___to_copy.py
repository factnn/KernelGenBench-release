import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['n_elements'],
)
@triton.jit
def _to_copy_kernel(
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
    tl.store(output_ptr + offsets, x, mask=mask)


def _to_copy(
    self: torch.Tensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    if layout is not None and layout != self.layout:
        raise RuntimeError("_to_copy: only same layout is supported")

    if device is not None and device != self.device:
        raise RuntimeError("_to_copy: only same device is supported")

    if pin_memory is not None and pin_memory:
        raise RuntimeError("_to_copy: pin_memory not supported")

    target_dtype = dtype if dtype is not None else self.dtype

    effective_memory_format = memory_format
    if effective_memory_format == torch.preserve_format:
        effective_memory_format = None

    # dtype change always produces contiguous output
    if target_dtype != self.dtype:
        input_tensor = self.contiguous()
        result = torch.empty_like(input_tensor, dtype=target_dtype)
    elif effective_memory_format == torch.contiguous_format:
        input_tensor = self.contiguous()
        result = torch.empty_like(input_tensor, dtype=target_dtype)
    elif effective_memory_format == torch.channels_last and self.ndim == 4:
        input_tensor = self.contiguous(memory_format=torch.channels_last)
        result = torch.empty_like(input_tensor, dtype=target_dtype)
    elif effective_memory_format == torch.channels_last_3d and self.ndim == 5:
        input_tensor = self.contiguous(memory_format=torch.channels_last_3d)
        result = torch.empty_like(input_tensor, dtype=target_dtype)
    else:
        # preserve_format with same dtype: clone-like behavior
        input_tensor = self.clone()
        result = torch.empty_like(input_tensor, dtype=target_dtype)

    n_elements = result.numel()
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _to_copy_kernel[grid](input_tensor, result, n_elements)

    return result


def _to_copy_out(
    self: torch.Tensor,
    *,
    non_blocking=False,
    memory_format=None,
    out: torch.Tensor,
):
    result = _to_copy(self, non_blocking=non_blocking, memory_format=memory_format)
    out.copy_(result)
    return out