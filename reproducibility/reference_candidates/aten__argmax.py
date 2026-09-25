import torch
import triton
import triton.language as tl


@triton.jit
def argmax_kernel_flat(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1: Each block finds local max and argmax in its chunk.
    Stores (max_val, global_argmax_index) pairs in the intermediate buffer.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=float('-inf'))
    vals_f32 = vals.to(tl.float32)

    block_best_val = tl.max(vals_f32, axis=0)
    block_best_local_idx = tl.argmax(vals_f32, axis=0)
    block_best_global_idx = block_start + block_best_local_idx

    tl.store(out_ptr + pid * 2, block_best_val)
    tl.store(out_ptr + pid * 2 + 1, block_best_global_idx)


@triton.jit
def argmax_reduce(
    in_ptr,
    out_ptr,
    n_results,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 2: Reduce intermediate (val, global_idx) pairs to final argmax.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_results

    val_offsets = offsets * 2
    idx_offsets = offsets * 2 + 1

    vals = tl.load(in_ptr + val_offsets, mask=mask, other=float('-inf'))
    global_idxs = tl.load(in_ptr + idx_offsets, mask=mask, other=0)

    best_pos = tl.argmax(vals, axis=0)
    weights = tl.where(tl.arange(0, BLOCK_SIZE) == best_pos, 1, 0)
    best_global_idx = tl.sum(global_idxs * weights)

    tl.store(out_ptr, best_global_idx)


@triton.jit
def argmax_kernel_dim_last(
    in_ptr,
    out_ptr,
    n_reduce,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute argmax along the last (contiguous) dimension.
    Each program handles one slice.
    """
    pid = tl.program_id(0)
    base_offset = pid * n_reduce

    best_val = float('-inf')
    best_idx = 0

    for block_start in range(0, n_reduce, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_reduce
        vals = tl.load(in_ptr + base_offset + offsets, mask=mask, other=float('-inf'))
        vals_f32 = vals.to(tl.float32)

        block_best_val = tl.max(vals_f32, axis=0)
        block_best_local_idx = tl.argmax(vals_f32, axis=0)

        update = block_best_val > best_val
        best_val = tl.where(update, block_best_val, best_val)
        best_idx = tl.where(update, block_start + block_best_local_idx, best_idx)

    tl.store(out_ptr + pid, best_idx)


def _select_block_size(n_reduce: int) -> int:
    """Select optimal BLOCK_SIZE based on reduction dimension size."""
    if n_reduce <= 64:
        return 64
    elif n_reduce <= 256:
        return 128
    elif n_reduce <= 1024:
        return 256
    elif n_reduce <= 4096:
        return 512
    else:
        return 1024


def argmax(self: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    """aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    ndim = self.ndim

    if self.numel() == 0:
        if dim is None:
            if keepdim:
                return torch.empty([1] * ndim, dtype=torch.int64, device=self.device)
            return torch.zeros([], dtype=torch.int64, device=self.device)
        dim_norm = dim if dim >= 0 else dim + ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim_norm] = 1
        else:
            out_shape.pop(dim_norm)
        return torch.empty(out_shape, dtype=torch.int64, device=self.device)

    if dim is None:
        result = _argmax_flat(self)
        if keepdim:
            result = result.reshape([1] * ndim)
        return result
    else:
        return _argmax_dim(self, dim, keepdim)


def _argmax_flat(x: torch.Tensor) -> torch.Tensor:
    """Compute argmax over the entire flattened tensor. Returns a scalar tensor."""
    x_flat = x.reshape(-1).contiguous()
    n_elements = x_flat.numel()

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (num_blocks,)

    intermediate = torch.empty(num_blocks * 2, dtype=torch.float32, device=x.device)
    argmax_kernel_flat[grid](
        x_flat, intermediate, n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )

    if num_blocks == 1:
        idx_val = intermediate[1].item()
        return torch.tensor(idx_val, dtype=torch.int64, device=x.device)

    out = torch.empty(1, dtype=torch.int64, device=x.device)
    REDUCE_BLOCK = triton.next_power_of_2(num_blocks)
    argmax_reduce[(1,)](
        intermediate, out, num_blocks, BLOCK_SIZE=REDUCE_BLOCK,
    )
    return out.reshape([])


def _argmax_dim(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    """Compute argmax along a specified dimension."""
    ndim = x.ndim
    dim = dim if dim >= 0 else dim + ndim

    out_shape = list(x.shape)
    n_reduce = x.shape[dim]
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)

    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)

    if n_reduce == 1:
        out.zero_()
        return out

    if ndim == 1:
        result_flat = _argmax_flat(x)
        if keepdim:
            return result_flat.reshape(out_shape)
        else:
            return result_flat

    # Permute so reduction dim becomes last, then make contiguous
    perm = list(range(ndim))
    perm.pop(dim)
    perm.append(dim)
    x_permuted = x.permute(*perm).contiguous()

    n_outer = x_permuted.numel() // n_reduce
    x_reshaped = x_permuted.reshape(n_outer, n_reduce)
    out_flat = out.reshape(-1)

    BLOCK_SIZE = _select_block_size(n_reduce)
    grid = (n_outer,)
    argmax_kernel_dim_last[grid](
        x_reshaped, out_flat, n_reduce, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def argmax_out(self: torch.Tensor, dim=None, keepdim=False, *, out: torch.Tensor) -> torch.Tensor:
    """aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    result = argmax(self, dim=dim, keepdim=keepdim)
    out.copy_(result)
    return out