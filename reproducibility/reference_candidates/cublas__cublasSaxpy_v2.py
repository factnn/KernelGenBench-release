import torch
import triton
import triton.language as tl


@triton.jit
def saxpy_kernel(n, alpha, x_ptr, incx, y_ptr, incy, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx
    y_idx = offsets * incy

    x = tl.load(x_ptr + x_idx, mask=mask)
    y = tl.load(y_ptr + y_idx, mask=mask)

    result = alpha * x + y
    tl.store(y_ptr + y_idx, result, mask=mask)


def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    saxpy_kernel[grid](n, alpha, x, incx, y, incy, BLOCK_SIZE=BLOCK_SIZE)
    return y