import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("convert_fp8")
@parametrize("input", [(1, 32), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("scale", [0.5, 1.0])
@parametrize("kv_dtype", ["fp8", "fp8_e4m3"])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_convert_fp8(input, scale, kv_dtype, dtype):
    # ===== Accuracy Test =====
    # Create inputs
    in_shape = input
    x = torch.randn(in_shape, device='cuda', dtype=dtype)

    # Create output tensors for baseline (FP8 stored as 8-bit)
    ref_out = torch.empty(in_shape, device='cuda', dtype=torch.uint8)
    # Clone for triton
    act_out = ref_out.clone()

    # Call baseline: kernelgenbench.baseline.convert_fp8(...)
    kernelgenbench.baseline.convert_fp8(ref_out, x, scale, kv_dtype)
    # Call triton:   kernelgenbench.triton.convert_fp8(...)
    kernelgenbench.triton.convert_fp8(act_out, x, scale, kv_dtype)

    # Compare: uint8 output, use exact comparison
    assert torch.equal(act_out, ref_out), f"Mismatch: max diff={( act_out.int() - ref_out.int()).abs().max()}"

    # ===== Performance Test =====
    # Skip small sizes for performance test
    numel = in_shape[0] * in_shape[1]
    if numel < 131072:
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn(in_shape, device='cuda', dtype=dtype)
    out_baseline = torch.empty(in_shape, device='cuda', dtype=torch.uint8)
    out_triton = torch.empty(in_shape, device='cuda', dtype=torch.uint8)

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.convert_fp8(out_baseline, x_bench, scale, kv_dtype),
        warmup=25, rep=100)

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.convert_fp8(out_triton, x_bench, scale, kv_dtype),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)