import triton
import triton.language as tl
import torch


def _get_configs():
    configs = []
    for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for num_warps in [4, 8, 16]:
            if num_warps * 32 <= block_size:
                configs.append(triton.Config(
                    {"BLOCK_SIZE": block_size},
                    num_warps=num_warps,
                    num_stages=2,
                ))
    return configs


@triton.autotune(
    configs=_get_configs(),
    key=["n_cols"],
)
@triton.jit
def _rms_norm_kernel(
    out_ptr,
    input_ptr,
    weight_ptr,
    n_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # Pass 1: Compute sum of squares using block-level reduction
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        var += x * x

    # Compute reciprocal sqrt of variance
    rstd = tl.math.rsqrt(tl.sum(var, axis=0) / n_cols + epsilon)

    # Pass 2: Normalize and apply weight
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row_start + offs, x * rstd * w, mask=mask)


def rms_norm(out, input, weight, epsilon):
    n_rows, n_cols = input.shape
    grid = (n_rows,)
    _rms_norm_kernel[grid](out, input, weight, n_cols, epsilon)