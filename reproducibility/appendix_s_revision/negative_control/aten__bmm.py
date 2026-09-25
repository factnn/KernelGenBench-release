import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    A_ptr += pid_batch.to(tl.int64) * stride_ab
    B_ptr += pid_batch.to(tl.int64) * stride_bb
    C_ptr += pid_batch.to(tl.int64) * stride_cb

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, 0.0, mask=c_mask)


def _compute_bmm(
    self: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype = None,
) -> torch.Tensor:
    B, M, K = self.shape
    _B, _K, N = mat2.shape

    if out_dtype is None:
        out_dtype = self.dtype

    output = torch.empty(B, M, N, device=self.device, dtype=out_dtype)

    a_stride_b, a_stride_m, a_stride_k = self.stride()
    b_stride_b, b_stride_k, b_stride_n = mat2.stride()
    c_stride_b, c_stride_m, c_stride_n = output.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        B,
    )

    bmm_kernel[grid](
        self, mat2, output,
        M, N, K,
        a_stride_b, a_stride_m, a_stride_k,
        b_stride_b, b_stride_k, b_stride_n,
        c_stride_b, c_stride_m, c_stride_n,
    )

    return output


def bmm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """aten::bmm(Tensor self, Tensor mat2) -> Tensor"""
    if self.dim() != 3 or mat2.dim() != 3:
        raise RuntimeError(
            f"bmm: Expected 3D tensors, got {self.dim()}D and {mat2.dim()}D"
        )
    if self.size(0) != mat2.size(0):
        raise RuntimeError(
            f"bmm: Batch size mismatch: {self.size(0)} vs {mat2.size(0)}"
        )
    if self.size(2) != mat2.size(1):
        raise RuntimeError(
            f"bmm: K dimension mismatch: {self.size(2)} vs {mat2.size(1)}"
        )

    return _compute_bmm(self, mat2)


def bmm_out(self: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"""
    result = _compute_bmm(self, mat2, out_dtype=out.dtype)
    out.copy_(result)
    return out


def bmm_dtype(self: torch.Tensor, mat2: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """aten::bmm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor"""
    if self.dim() != 3 or mat2.dim() != 3:
        raise RuntimeError(
            f"bmm: Expected 3D tensors, got {self.dim()}D and {mat2.dim()}D"
        )
    if self.size(0) != mat2.size(0):
        raise RuntimeError(
            f"bmm: Batch size mismatch: {self.size(0)} vs {mat2.size(0)}"
        )
    if self.size(2) != mat2.size(1):
        raise RuntimeError(
            f"bmm: K dimension mismatch: {self.size(2)} vs {mat2.size(1)}"
        )

    return _compute_bmm(self, mat2, out_dtype=out_dtype)


def bmm_dtype_out(self: torch.Tensor, mat2: torch.Tensor, out_dtype: torch.dtype, *, out: torch.Tensor) -> torch.Tensor:
    """aten::bmm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *, Tensor(a!) out) -> Tensor(a!)"""
    result = _compute_bmm(self, mat2, out_dtype=out_dtype)
    out.copy_(result)
    return out