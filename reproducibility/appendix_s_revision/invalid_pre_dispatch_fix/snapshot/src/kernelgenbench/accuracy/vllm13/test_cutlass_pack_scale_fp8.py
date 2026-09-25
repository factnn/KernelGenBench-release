import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("cutlass_pack_scale_fp8")
@parametrize("shape", [
    (1, 4),
    (4, 4),
    (8, 8),
    (16, 16),
    (4, 16),
    (32, 8),
    (64, 16),
    (128, 32),
])
def test_accuracy_cutlass_pack_scale_fp8(shape):
    # ===== Accuracy Test =====
    rows, cols = shape[0], shape[1]
    scales = torch.randn(rows, cols, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    ref_out = kernelgenbench.baseline.cutlass_pack_scale_fp8(scales)
    act_out = kernelgenbench.triton.cutlass_pack_scale_fp8(scales)

    assert torch.equal(act_out, ref_out), f"Mismatch: max diff={(act_out.float() - ref_out.float()).abs().max()}"

    # ===== Performance Test =====
    if rows * cols < 64:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.cutlass_pack_scale_fp8(scales),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.cutlass_pack_scale_fp8(scales),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
