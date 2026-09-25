import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple contiguous-to-contiguous element-wise copy."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def strided_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    ndim,
    # Shapes of the OUTPUT (contiguous), padded to 6 dims
    shape0, shape1, shape2, shape3, shape4, shape5,
    # Strides of the SOURCE (strided), padded to 6 dims
    src_stride0, src_stride1, src_stride2, src_stride3, src_stride4, src_stride5,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy from a strided source to a contiguous destination.

    For a contiguous output with shape [d0,...,d5], linear index:
      linear = ((((i0*d1 + i1)*d2 + i2)*d3 + i3)*d4 + i4)*d5 + i5

    Source offset = i0*src_s0 + i1*src_s1 + ... + i5*src_s5

    We decompose linear index using Horner's method.
    Shapes are padded on the LEFT with 1s, strides with 0s.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements

    remaining = offsets
    src_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Process dims 5 down to 0 using Horner's method
    # For padded dims (ndim < 6), shape is 1 so idx=0 always.

    idx5 = remaining % shape5
    remaining = remaining // shape5
    src_offsets += idx5 * src_stride5

    idx4 = remaining % shape4
    remaining = remaining // shape4
    src_offsets += idx4 * src_stride4

    idx3 = remaining % shape3
    remaining = remaining // shape3
    src_offsets += idx3 * src_stride3

    idx2 = remaining % shape2
    remaining = remaining // shape2
    src_offsets += idx2 * src_stride2

    idx1 = remaining % shape1
    remaining = remaining // shape1
    src_offsets += idx1 * src_stride1

    idx0 = remaining % shape0
    remaining = remaining // shape0
    src_offsets += idx0 * src_stride0

    vals = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def channels_last_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    src_stride_n, src_stride_c, src_stride_h, src_stride_w,
    dst_stride_n, dst_stride_h, dst_stride_w, dst_stride_c,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Convert to channels_last (NHWC) memory format.
    Output linear index to NHWC: dst[n, h, w, c] -> n*N_stride + h*H_stride + w*W_stride + c*C_stride
    Source index from NCHW layout: src[n, c, h, w] -> n*src_n + c*src_c + h*src_h + w*src_w
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    CW = C * W
    CHW = C * H * W

    n = offsets // CHW
    rem_hw = offsets % CHW
    h = rem_hw // CW
    rem_cw = rem_hw % CW
    w = rem_cw // C
    c = rem_cw % C

    src_idx = n * src_stride_n + c * src_stride_c + h * src_stride_h + w * src_stride_w
    dst_idx = n * dst_stride_n + h * dst_stride_h + w * dst_stride_w + c * dst_stride_c

    vals = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def channels_last_3d_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    src_stride_n, src_stride_c, src_stride_d, src_stride_h, src_stride_w,
    dst_stride_n, dst_stride_d, dst_stride_h, dst_stride_w, dst_stride_c,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Convert to channels_last_3d (NDHWC) memory format.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    W_vol = W
    HW = H * W
    DHW = D * H * W
    CDHW = C * D * H * W

    n = offsets // CDHW
    rem_dhw = offsets % CDHW
    d = rem_dhw // DHW
    rem_hw = rem_dhw % DHW
    h = rem_hw // HW
    rem_w = rem_hw % HW
    w = rem_w // W_vol
    c = rem_w % W_vol

    src_idx = (n * src_stride_n + c * src_stride_c +
               d * src_stride_d + h * src_stride_h + w * src_stride_w)
    dst_idx = (n * dst_stride_n + d * dst_stride_d +
               h * dst_stride_h + w * dst_stride_w + c * dst_stride_c)

    vals = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, vals, mask=mask)


def _run_copy_kernel(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Run simple contiguous copy kernel."""
    n_elements = src.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](src, dst, n_elements)


def _run_strided_copy_kernel(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Run strided copy kernel for non-contiguous source."""
    n_elements = src.numel()
    ndim = src.ndim

    # We generate a padded version of shapes and strides for up to 6 dims
    # Padding is on the LEFT: for ndim=N, the original dims are at positions (6-N) through 5
    shapes = [1] * 6
    strides = [0] * 6
    orig_shapes = list(src.shape)
    orig_strides = list(src.stride())
    offset = 6 - ndim
    for i in range(ndim):
        shapes[offset + i] = orig_shapes[i]
        strides[offset + i] = orig_strides[i]

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    strided_copy_kernel[grid](
        src, dst,
        n_elements, ndim,
        shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5],
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
    )


def _make_contiguous(self: torch.Tensor, memory_format: int = 0) -> torch.Tensor:
    """
    Core implementation of contiguous using Triton kernels.

    Args:
        memory_format: 0=contiguous, 1=channels_last, 2=channels_last_3d, 3=preserve
    """
    if self.ndim == 0:
        return self.clone()

    n_elements = self.numel()

    if memory_format == 1:  # channels_last
        if self.ndim == 4:
            output = torch.empty_like(self, memory_format=torch.channels_last)
            N, C, H, W = self.shape
            src_sn, src_sc, src_sh, src_sw = self.stride()
            dst_sn, dst_sh, dst_sw, dst_sc = output.stride()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            channels_last_kernel[grid](
                self, output,
                n_elements,
                src_sn, src_sc, src_sh, src_sw,
                dst_sn, dst_sh, dst_sw, dst_sc,
                N, C, H, W,
            )
            return output
        else:
            output = torch.empty_like(self)
            _run_strided_copy_kernel(self, output)
            return output

    elif memory_format == 2:  # channels_last_3d
        if self.ndim == 5:
            output = torch.empty_like(self, memory_format=torch.channels_last_3d)
            N, C, D, H, W = self.shape
            src_sn, src_sc, src_sd, src_sh, src_sw = self.stride()
            dst_sn, dst_sd, dst_sh, dst_sw, dst_sc = output.stride()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            channels_last_3d_kernel[grid](
                self, output,
                n_elements,
                src_sn, src_sc, src_sd, src_sh, src_sw,
                dst_sn, dst_sd, dst_sh, dst_sw, dst_sc,
                N, C, D, H, W,
            )
            return output
        else:
            output = torch.empty_like(self)
            _run_strided_copy_kernel(self, output)
            return output

    else:  # memory_format == 0 (contiguous)
        output = torch.empty_like(self)

        if self.is_contiguous():
            _run_copy_kernel(self, output)
        else:
            _run_strided_copy_kernel(self, output)

        return output


def contiguous(self: torch.Tensor, memory_format: int = 0) -> torch.Tensor:
    """
    实现 aten::contiguous

    Args:
        self: 输入 tensor
        memory_format: 0 = torch.contiguous_format (C-contiguous)
                      1 = torch.channels_last (NHWC for 4D)
                      2 = torch.channels_last_3d (for 5D)
                      3 = torch.preserve_format

    Returns:
        具有指定 memory format 的连续 tensor
    """
    # If already in the right format, return self
    if memory_format == 0:  # contiguous_format
        if self.is_contiguous():
            return self
    elif memory_format == 1:  # channels_last
        if self.ndim == 4 and self.is_contiguous(memory_format=torch.channels_last):
            return self
    elif memory_format == 2:  # channels_last_3d
        if self.ndim == 5 and self.is_contiguous(memory_format=torch.channels_last_3d):
            return self
    elif memory_format == 3:  # preserve_format
        # preserve_format: return self if already contiguous in its "natural" format
        if self.is_contiguous():
            return self
        # For preserve on non-contiguous, suggest memory format
        suggested = torch._C._autograd._suggest_memory_format(self)
        if suggested == torch.channels_last:
            memory_format = 1
        elif suggested == torch.channels_last_3d:
            memory_format = 2
        else:
            memory_format = 0

    return _make_contiguous(self, memory_format)