import torch
import triton
import triton.language as tl


@triton.jit
def index_select_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    outer_dim,
    dim_size,
    inner_dim,
    index_len,
    input_outer_stride,
    input_dim_stride,
    input_inner_stride,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for index_select: gathers elements from input along dimension `dim`
    at positions specified by `index` and writes them contiguously to output.

    The output has layout: (outer_dim, index_len, inner_dim)

    Each program block processes BLOCK_SIZE output elements.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decompose 1D output offset into (outer_idx, idx_pos, inner_idx)
    idx_inner = index_len * inner_dim
    outer_idx = offsets // idx_inner
    remainder = offsets % idx_inner
    idx_pos = remainder // inner_dim
    inner_idx = remainder % inner_dim

    # Lookup the source dimension index
    src_dim_idx = tl.load(index_ptr + idx_pos, mask=mask)

    # Compute input offset: outer_idx * outer_stride + src_dim_idx * dim_stride + inner_idx * inner_stride
    input_offsets = (
        outer_idx.to(tl.int64) * input_outer_stride
        + src_dim_idx.to(tl.int64) * input_dim_stride
        + inner_idx.to(tl.int64) * input_inner_stride
    )

    # Gather from input, store to output
    value = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + offsets, value, mask=mask)


def index_select(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
) -> torch.Tensor:
    """实现 aten::index_select

    Gathers values along dimension `dim` from `self` at positions specified by `index`.
    Equivalent to PyTorch's torch.index_select(self, dim, index).
    """
    ndim = self.dim()
    if dim < 0:
        dim += ndim

    self_contig = self.contiguous()
    index_contig = index.contiguous()

    outer_sizes = self.shape[:dim]
    inner_sizes = self.shape[dim + 1:]
    index_len = index.numel()

    outer_dim = 1
    for s in outer_sizes:
        outer_dim *= s
    inner_dim = 1
    for s in inner_sizes:
        inner_dim *= s
    dim_size = self.shape[dim]

    output_shape = outer_sizes + (index_len,) + inner_sizes
    output = torch.empty(output_shape, dtype=self.dtype, device=self.device)
    total_elements = output.numel()

    if total_elements == 0:
        return output

    # For contiguous tensor, element at (outer, dim_val, inner):
    #   offset = outer * (dim_size * inner_dim) + dim_val * inner_dim + inner * 1
    outer_stride = dim_size * inner_dim if outer_dim > 1 else 0
    dim_stride = inner_dim if inner_dim > 0 else 1
    inner_stride = 1 if inner_dim > 0 else 0

    BLOCK_SIZE = 512
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    index_select_kernel[grid](
        self_contig,
        index_contig,
        output,
        outer_dim,
        dim_size,
        inner_dim,
        index_len,
        outer_stride,
        dim_stride,
        inner_stride,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def index_select_out(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """实现 aten::index_select.out"""
    result = index_select(self, dim, index)
    out.copy_(result)
    return out


def index_select_dimname(
    self: torch.Tensor,
    dim: str,
    index: torch.Tensor,
) -> torch.Tensor:
    """实现 aten::index_select.dimname"""
    names = self.names
    dim_idx = 0
    for i, name in enumerate(names):
        if name == dim:
            dim_idx = i
            break
    return index_select(self, dim_idx, index)


def index_select_dimname_out(
    self: torch.Tensor,
    dim: str,
    index: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """实现 aten::index_select.dimname_out"""
    result = index_select_dimname(self, dim, index)
    out.copy_(result)
    return out