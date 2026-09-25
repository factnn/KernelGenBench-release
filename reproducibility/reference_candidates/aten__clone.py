import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def clone_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


def clone(self: torch.Tensor, *, memory_format=None) -> torch.Tensor:
    """实现 aten::clone"""
    if memory_format == torch.contiguous_format:
        output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    elif memory_format == torch.preserve_format or memory_format is None:
        output = torch.empty_strided(self.shape, self.stride(),
                                     dtype=self.dtype, device=self.device)
    else:
        output = torch.empty(self.shape, dtype=self.dtype, device=self.device)

    n_elements = output.numel()

    if self.is_contiguous() and output.is_contiguous():
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        clone_kernel[grid](self, output, n_elements)
    elif memory_format == torch.contiguous_format:
        # Non-contiguous input -> contiguous output
        _clone_strided_to_contiguous(self, output)
    else:
        # preserve_format: both have same strides
        _clone_strided_impl(self, output)

    return output


def _clone_strided_to_contiguous(src: torch.Tensor, dst: torch.Tensor):
    """Copy from non-contiguous src to contiguous dst."""
    n_elements = src.numel()
    shape = src.shape
    src_strides = src.stride()
    ndim = len(shape)

    if ndim == 0:
        n_elements = 1
        grid = lambda meta: (1,)
        clone_kernel[grid](src.reshape(1), dst.reshape(1), 1)
        return

    if ndim == 1:
        _strided_to_contig_impl_1d(src, dst, shape[0], src_strides[0])
    elif ndim == 2:
        _strided_to_contig_impl_2d(src, dst, shape[0], shape[1],
                                   src_strides[0], src_strides[1])
    elif ndim == 3:
        _strided_to_contig_impl_3d(src, dst, shape[0], shape[1], shape[2],
                                   src_strides[0], src_strides[1], src_strides[2])
    elif ndim == 4:
        _strided_to_contig_impl_4d(src, dst, shape[0], shape[1], shape[2], shape[3],
                                   src_strides[0], src_strides[1], src_strides[2], src_strides[3])
    elif ndim == 5:
        _strided_to_contig_impl_5d(src, dst, shape[0], shape[1], shape[2], shape[3], shape[4],
                                   src_strides[0], src_strides[1], src_strides[2], src_strides[3], src_strides[4])
    else:
        _strided_to_contig_impl_nd(src, dst)


def _clone_strided_impl(src: torch.Tensor, dst: torch.Tensor):
    """Copy from src to dst with same strides (preserve_format)."""
    if src.is_contiguous() and dst.is_contiguous():
        n_elements = src.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        clone_kernel[grid](src, dst, n_elements)
        return

    shape = src.shape
    src_strides = src.stride()
    dst_strides = dst.stride()
    ndim = len(shape)

    if ndim == 0:
        n_elements = 1
        grid = lambda meta: (1,)
        clone_kernel[grid](src.reshape(1), dst.reshape(1), 1)
        return

    if ndim == 1:
        _preserve_impl_1d(src, dst, shape[0], src_strides[0], dst_strides[0])
    elif ndim == 2:
        _preserve_impl_2d(src, dst, shape[0], shape[1],
                          src_strides[0], src_strides[1],
                          dst_strides[0], dst_strides[1])
    else:
        _strided_general_impl(src, dst, shape, src_strides, dst_strides)


# --- 1D kernels ---

@triton.jit
def _strided_to_contig_1d_kernel(
    src_ptr, dst_ptr,
    dim0, src_s0,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    src_idx = offsets * src_s0
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


@triton.jit
def _preserve_1d_kernel(
    src_ptr, dst_ptr,
    dim0, src_s0, dst_s0,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    src_idx = offsets * src_s0
    dst_idx = offsets * dst_s0
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, val, mask=mask)


# --- 2D kernels ---

@triton.jit
def _strided_to_contig_2d_kernel(
    src_ptr, dst_ptr,
    dim0, dim1, src_s0, src_s1,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // dim1
    col = offsets % dim1
    src_idx = row * src_s0 + col * src_s1
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


@triton.jit
def _preserve_2d_kernel(
    src_ptr, dst_ptr,
    dim0, dim1, src_s0, src_s1, dst_s0, dst_s1,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // dim1
    col = offsets % dim1
    src_idx = row * src_s0 + col * src_s1
    dst_idx = row * dst_s0 + col * dst_s1
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, val, mask=mask)


# --- 3D kernels ---

@triton.jit
def _strided_to_contig_3d_kernel(
    src_ptr, dst_ptr,
    dim0, dim1, dim2, src_s0, src_s1, src_s2,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    d2 = idx % dim2
    idx = idx // dim2
    d1 = idx % dim1
    d0 = idx // dim1
    src_idx = d0 * src_s0 + d1 * src_s1 + d2 * src_s2
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


# --- 4D kernels ---

@triton.jit
def _strided_to_contig_4d_kernel(
    src_ptr, dst_ptr,
    dim0, dim1, dim2, dim3,
    src_s0, src_s1, src_s2, src_s3,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    d3 = idx % dim3
    idx = idx // dim3
    d2 = idx % dim2
    idx = idx // dim2
    d1 = idx % dim1
    d0 = idx // dim1
    src_idx = d0 * src_s0 + d1 * src_s1 + d2 * src_s2 + d3 * src_s3
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


# --- 5D kernels ---

@triton.jit
def _strided_to_contig_5d_kernel(
    src_ptr, dst_ptr,
    dim0, dim1, dim2, dim3, dim4,
    src_s0, src_s1, src_s2, src_s3, src_s4,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    d4 = idx % dim4
    idx = idx // dim4
    d3 = idx % dim3
    idx = idx // dim3
    d2 = idx % dim2
    idx = idx // dim2
    d1 = idx % dim1
    d0 = idx // dim1
    src_idx = d0 * src_s0 + d1 * src_s1 + d2 * src_s2 + d3 * src_s3 + d4 * src_s4
    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


# --- General ND (fallback) ---

def _strided_general_impl(src, dst, shape, src_strides, dst_strides):
    n_elements = src.numel()
    ndim = len(shape)
    shape_t = torch.tensor(list(shape), dtype=torch.int64, device='cuda')
    src_stride_t = torch.tensor(list(src_strides), dtype=torch.int64, device='cuda')
    dst_stride_t = torch.tensor(list(dst_strides), dtype=torch.int64, device='cuda')

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _strided_general_kernel[grid](
        src, dst, shape_t, src_stride_t, dst_stride_t,
        ndim, n_elements,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _strided_general_kernel(
    src_ptr, dst_ptr,
    shape_ptr, src_stride_ptr, dst_stride_ptr,
    ndim, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    elem_idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = elem_idx < n_elements

    remaining = elem_idx.to(tl.int64)
    src_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    for d in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + d)
        src_stride = tl.load(src_stride_ptr + d)
        dst_stride = tl.load(dst_stride_ptr + d)

        coord = remaining % dim_size
        remaining = remaining // dim_size

        src_offset += coord.to(tl.int64) * src_stride
        dst_offset += coord.to(tl.int64) * dst_stride

    val = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset, val, mask=mask)


def _strided_to_contig_impl_nd(src, dst):
    n_elements = src.numel()
    shape = src.shape
    src_strides = src.stride()
    ndim = len(shape)
    shape_t = torch.tensor(list(shape), dtype=torch.int64, device='cuda')
    src_stride_t = torch.tensor(list(src_strides), dtype=torch.int64, device='cuda')

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _strided_to_contig_general_kernel[grid](
        src, dst, shape_t, src_stride_t, ndim, n_elements,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _strided_to_contig_general_kernel(
    src_ptr, dst_ptr,
    shape_ptr, src_stride_ptr,
    ndim, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    elem_idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = elem_idx < n_elements

    remaining = elem_idx.to(tl.int64)
    src_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    for d in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + d)
        src_stride = tl.load(src_stride_ptr + d)

        coord = remaining % dim_size
        remaining = remaining // dim_size

        src_offset += coord.to(tl.int64) * src_stride

    val = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(dst_ptr + elem_idx, val, mask=mask)


# --- Dispatch helpers ---

def _strided_to_contig_impl_1d(src, dst, d0, s0):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _strided_to_contig_1d_kernel[grid](src, dst, d0, s0, n, BLOCK_SIZE=256)

def _strided_to_contig_impl_2d(src, dst, d0, d1, s0, s1):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _strided_to_contig_2d_kernel[grid](src, dst, d0, d1, s0, s1, n, BLOCK_SIZE=256)

def _strided_to_contig_impl_3d(src, dst, d0, d1, d2, s0, s1, s2):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _strided_to_contig_3d_kernel[grid](src, dst, d0, d1, d2, s0, s1, s2, n,
                                       BLOCK_SIZE=256)

def _strided_to_contig_impl_4d(src, dst, d0, d1, d2, d3, s0, s1, s2, s3):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _strided_to_contig_4d_kernel[grid](src, dst, d0, d1, d2, d3,
                                       s0, s1, s2, s3, n,
                                       BLOCK_SIZE=256)

def _strided_to_contig_impl_5d(src, dst, d0, d1, d2, d3, d4,
                                s0, s1, s2, s3, s4):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _strided_to_contig_5d_kernel[grid](src, dst, d0, d1, d2, d3, d4,
                                       s0, s1, s2, s3, s4, n,
                                       BLOCK_SIZE=256)

def _preserve_impl_1d(src, dst, d0, s0, ds0):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _preserve_1d_kernel[grid](src, dst, d0, s0, ds0, n, BLOCK_SIZE=256)

def _preserve_impl_2d(src, dst, d0, d1, s0, s1, ds0, ds1):
    n = src.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _preserve_2d_kernel[grid](src, dst, d0, d1, s0, s1, ds0, ds1, n,
                              BLOCK_SIZE=256)


def clone_out(self: torch.Tensor, *, out: torch.Tensor, memory_format=None) -> torch.Tensor:
    """实现 aten::clone.out"""
    result = clone(self, memory_format=memory_format)
    out.copy_(result)
    return out