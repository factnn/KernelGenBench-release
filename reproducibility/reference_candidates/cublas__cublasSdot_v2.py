import torch
import triton
import triton.language as tl


@triton.jit
def sdot_kernel(
    n,
    x_ptr,
    incx,
    y_ptr,
    incy,
    partial_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx
    y_idx = offsets * incy

    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)
    y_val = tl.load(y_ptr + y_idx, mask=mask, other=0.0).to(tl.float32)

    partial = tl.sum(x_val * y_val)

    tl.store(partial_ptr + pid, partial)


@triton.jit
def reduce_kernel(
    partial_ptr,
    result_ptr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks

    vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(vals.to(tl.float32))

    if tl.program_id(0) == 0:
        tl.store(result_ptr, total)


def cublasSdot_v2(n, x, incx, y, incy, result):
    n = int(n)
    incx = int(incx)
    incy = int(incy)

    if n == 0:
        result.zero_()
        return result

    if n <= 64:
        BLOCK_SIZE = 32
    elif n <= 256:
        BLOCK_SIZE = 128
    elif n <= 1024:
        BLOCK_SIZE = 256
    elif n <= 4096:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    partial = torch.empty(num_blocks, dtype=torch.float32, device=x.device)

    grid = (num_blocks,)
    sdot_kernel[grid](n, x, incx, y, incy, partial, BLOCK_SIZE=BLOCK_SIZE)

    reduce_blk = min(1024, triton.next_power_of_2(num_blocks))
    reduce_grid = (1,)
    reduce_kernel[reduce_grid](partial, result, num_blocks, BLOCK_SIZE=reduce_blk)

    return result