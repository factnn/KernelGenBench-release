import importlib
import itertools

import torch
from pydantic import BaseModel
from typing import Callable, List, Optional, Union, Dict


import random
import numpy as np

# import kernelgenbench

# from .conftest import QUICK_MODE, TO_CPU
# from .config import QUICK_MODE, TO_CPU
from sandbox.config import QUICK_MODE, TO_CPU

import os
import math
UPCAST = os.environ.get("KERNELGENBENCH_UPCAST", "1") == "1"

fp64_is_supported = True

import torch

# ==============================================================================
# Evaluation-policy switches
# ==============================================================================
# These switches exist so that the *published* protocol can be re-run under
# alternative validation policies without changing the default behaviour.  With
# every switch left at its default the module reproduces the paper protocol
# bit-for-bit; the alternatives are used by the analyses in scripts/analyze/
# (tolerance sensitivity and held-out-shape generalization).
#
#   KGB_ATOL_MODE    scaled | const | sqrt     (default: scaled)
#                    scaled -> atol = 1e-4 * D_reduce            (paper protocol)
#                    sqrt   -> atol = 1e-4 * sqrt(D_reduce)
#                    const  -> atol = 1e-4
#   KGB_HELDOUT      "1" to append held-out shapes/strides that are never
#                    exposed to the generator or the agent (default: off)
#   KGB_SEED_OFFSET  integer added to the verification seed, so that held-out
#                    runs also see different input *values* (default: 0)
# ==============================================================================

ATOL_MODE = os.environ.get("KGB_ATOL_MODE", "scaled").strip().lower()
HELDOUT_MODE = os.environ.get("KGB_HELDOUT", "0") == "1"
SEED_OFFSET = int(os.environ.get("KGB_SEED_OFFSET", "0"))
# When KGB_AUDIT points at a directory, every comparison that passes the
# published tolerance also records how much tolerance it actually needed, and
# ``reverify_corpus.py --audit-dir`` turns that record into the pass/fail
# outcome for a whole family of tolerance rules at no extra GPU cost.
AUDIT_PATH = os.environ.get("KGB_AUDIT", "").strip()

if ATOL_MODE not in ("scaled", "const", "sqrt"):
    raise ValueError(f"Unknown KGB_ATOL_MODE: {ATOL_MODE!r}")


def audit(record: dict) -> None:
    """Append one JSON record to the audit log, if auditing is enabled."""
    if not AUDIT_PATH:
        return
    import json as _json
    record["pid"] = os.getpid()
    try:
        with open(AUDIT_PATH, "a") as handle:
            handle.write(_json.dumps(record) + "\n")
    except Exception:
        pass

def heldout_params(default, extra):
    """Parameter grid used by a test: the published values, plus the held-out
    values that KGB_HELDOUT appends and that are never exposed to the generator
    or to the agent."""
    return list(default) + list(extra) if HELDOUT_MODE else list(default)


class CustomBenchmarkResult(BaseModel):
    ref_time: float
    res_time: float
    speedup: float
    params: Union[dict, str] = None
    def to_dict(self):
        return {
            "ref_time": self.ref_time,
            "res_time": self.res_time,
            "speedup": self.speedup,
            "params": self.params,
        }

class VerifyResult(BaseModel):
    op_name: str = None
    success: Optional[bool] = None
    traceback: Optional[str] = None
    params: Optional[dict] = None
    speedup: Optional[List[Union[CustomBenchmarkResult, Dict]]] = None
    info: Optional[dict] = None
    code: Optional[str] = None
    test_func: Optional[str] = None

RESOLUTION = {
    torch.bool: 0,
    torch.uint8: 0,
    torch.uint16: 0,
    torch.uint32: 0,
    torch.uint64: 0,
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 1e-3,
    # torch.float32: 1.3e-6,
    # torch.float64: 1e-6,
    torch.float32: 1e-5,
    torch.float64: 1e-5,
    torch.bfloat16: 0.016,
    torch.complex32: 1e-3,
    torch.complex64: 1.3e-6,
    torch.complex128: 1e-6,
}


def effective_atol(reduce_dim: int) -> float:
    """Absolute tolerance for a comparison over ``reduce_dim`` accumulated terms.

    ``scaled`` is the protocol used for every number reported in the paper:
    ``atol = 1e-4 * D_reduce``, which grows linearly with the number of
    accumulated terms.  The alternatives are used to test whether the reported
    pass rates are sensitive to that choice.
    """
    d = max(int(reduce_dim), 1)
    if ATOL_MODE == "const":
        return 1e-4
    if ATOL_MODE == "sqrt":
        return 1e-4 * math.sqrt(d)
    return 1e-4 * d


def _record_tolerance_slack(res, ref, rtol, reduce_dim, dtype):
    """Record how much tolerance a passing comparison actually consumed.

    ``torch.testing.assert_close`` accepts a pair when
    ``|res - ref| <= atol + rtol * |ref|`` elementwise.  For a passing
    comparison the smallest ``atol`` that would still accept the pair is
    ``max(|res - ref| - rtol * |ref|)``; we log that value together with the
    reduction length, so that any ``atol = f(D_reduce)`` rule can be evaluated
    afterwards without re-running the kernel.
    """
    if not AUDIT_PATH or not (dtype.is_floating_point or dtype.is_complex):
        return
    try:
        diff = (res - ref).abs()
        mag = ref.abs()
        both_nan = torch.isnan(diff) & torch.isnan(mag)
        if bool(both_nan.any()):
            zero = torch.zeros_like(diff)
            diff = torch.where(both_nan, zero, diff)
            mag = torch.where(both_nan, zero, mag)
        slack = diff - rtol * mag
        slack = slack[~torch.isnan(slack)]
        max_slack = float(slack.max()) if slack.numel() else 0.0
        max_diff = float(diff[~torch.isnan(diff)].max()) if diff.numel() else 0.0
        ref_abs_max = float(mag[~torch.isnan(mag)].max()) if mag.numel() else 0.0
    except Exception:
        return
    audit({
        "kind": "tolerance",
        "reduce_dim": int(reduce_dim),
        "rtol": float(rtol),
        "dtype": str(dtype),
        "max_slack": max_slack,
        # ``max_diff`` and ``ref_abs_max`` bound the same comparison normwise,
        # which is what a scale-aware rule uses instead of a per-element one.
        "max_diff": max_diff,
        "ref_abs_max": ref_abs_max,
    })


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = effective_atol(reduce_dim)
    rtol = RESOLUTION[dtype]
    if AUDIT_PATH:
        torch.testing.assert_close(res, ref, atol=atol, rtol=rtol,
                                   equal_nan=equal_nan)
        _record_tolerance_slack(res, ref, rtol, reduce_dim, dtype)
        return
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)


def assert_equal(res, ref, equal_nan=False):
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)



def SkipVersion(module_name, skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">"), f"Invalid comparison operator: {cmp}"
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from skip_pattern.")

    try:
        module = importlib.import_module(module_name)
        version = module.__version__
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1] if QUICK_MODE else [1, 16, 64, 1000]
sizes_2d_nr = [1] if QUICK_MODE else [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SPECIAL_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 1280), (16, 7, 57, 32, 29)]
)
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 40999, 3)]
REDUCTION_SMALL_SHAPES = (
    [(1, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 2560, 3)]
)
STACK_SHAPES = [
    [(16,), (16,)],
    [(16, 256), (16, 256)],
    [(20, 320, 15), (20, 320, 15), (20, 320, 15)],
]
CONTIGUOUS_SHAPE_STRIDES_1D = [
    ((1,), (1,)),
    ((1024,), (1,)),
    ((1000000,), (1,)),
]
DILATED_SHAPE_STRIDES_1D = [
    ((1,), (2,)),
    ((1024,), (2,)),
    ((1000000,), (2,)),
]
CONTIGUOUS_SHAPE_STRIDES_2D = [
    ((1, 1024), (1024, 1)),
    ((10000, 128), (128, 1)),
]
TRANSPOSED_SHAPE_STRIDES_2D = [
    ((1024, 1), (1, 1024)),
    ((128, 10000), (1, 128)),
]
CONTIGUOUS_SHAPE_STRIDES_3D = [
    ((20, 320, 15), (4800, 15, 1)),
    ((200, 40999, 3), (122997, 3, 1)),
]
TRANSPOSED_SHAPE_STRIDES_3D = [
    ((320, 20, 15), (15, 4800, 1)),
    ((3, 40999, 200), (1, 3, 122997)),
]
SHAPE_STRIDES = (
    CONTIGUOUS_SHAPE_STRIDES_1D
    + DILATED_SHAPE_STRIDES_1D
    + CONTIGUOUS_SHAPE_STRIDES_2D
    + TRANSPOSED_SHAPE_STRIDES_2D
    + CONTIGUOUS_SHAPE_STRIDES_3D
    + TRANSPOSED_SHAPE_STRIDES_3D
)

IRREGULAR_SHAPE_STRIDES = [((10, 10, 10, 10, 10), (1, 10000, 23, 399, 1024))]

UPSAMPLE_SHAPES = [
    (32, 16, 128, 128),
    (15, 37, 256, 256),
    (3, 5, 127, 127),
    (128, 192, 42, 51),
    (3, 7, 1023, 1025),
]


KRON_SHAPES = [
    [(), (2, 3)],
    [(2, 3), ()],
    [(0, 3), (2, 3)],
    [(2, 3), (0,)],
    [(0,), (0,)],
    [(), ()],
    [(1,), (2,)],
    [(2,), (3,)],
    [(2, 2), (3, 3)],
    [(1, 2, 3), (2, 3, 4)],
    [(1,), (2, 2)],
    [(1, 2), (3, 4, 5)],
    [(2,), (3, 4, 5, 6)],
    [(2, 3, 4), (1,)],
    [(5, 5), (4, 4)],
    [(3, 3, 3), (2, 2, 2)],
    [(4, 4, 4, 4), (2, 2, 2, 2)],
    [(2, 3, 4), (3, 4, 5)],
    [(1, 3, 5), (2, 4, 6)],
    [(2, 4, 6, 8), (1, 3, 5, 7)],
    [(1, 3), (1, 4)],
    [(1, 1, 3), (1, 1, 2)],
    [(2, 1, 4), (3, 1, 5)],
    [(2, 2, 2, 2, 2), (1, 1, 1, 1, 1)],
    [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)],
    [(1,), (1,)],
    [(10,), (10,)],
    [(2, 3), (3, 2)],
    [(3, 3), (3, 3)],
    [(1, 1, 1), (2, 2, 2)],
]
# ==============================================================================
# Held-out shape / stride grids (KGB_HELDOUT=1)
# ==============================================================================
# These grids are *never* exposed to the generator or to the agent: they are not
# part of the published test suite, the prompts, or the verification CLI used
# during generation.  They mirror the structure of the published grids (same
# per-slot tensor rank and layout family) but use different sizes, so that a
# kernel which is only correct on the published shapes is detected.  With
# KGB_HELDOUT unset nothing in this block is appended and the published test
# suite is reproduced exactly.
HELDOUT_UT_SHAPES_1D = [(n,) for n in [49, 63, 145, 529, 2065, 4097]]
HELDOUT_UT_SHAPES_2D = list(itertools.product([3, 7, 999], [2, 31, 127, 2001]))
HELDOUT_POINTWISE_SHAPES = [(), (7,), (511, 257), (7, 17, 31), (5, 7, 11, 13), (2, 3, 4, 5, 6)]
HELDOUT_SPECIAL_SHAPES = [(7,), (2047, 2047), (7, 17, 31), (5, 7, 11, 13), (2, 3, 4, 5, 6)]
HELDOUT_DISTRIBUTION_SHAPES = [(7, 17, 31)]
HELDOUT_REDUCTION_SHAPES = [(3, 5), (2048, 129), (151, 8191, 5)]
HELDOUT_REDUCTION_SMALL_SHAPES = [(3, 5), (2048, 129), (151, 1021, 5)]
HELDOUT_STACK_SHAPES = [
    [(7,), (7,)],
    [(7, 255), (7, 255)],
    [(7, 17, 31), (7, 17, 31), (7, 17, 31)],
]
HELDOUT_SHAPE_STRIDES = [
    # 1D contiguous / dilated
    ((7,), (1,)),
    ((4096,), (1,)),
    ((255,), (3,)),
    ((4096,), (2,)),
    # 2D contiguous / transposed
    ((33, 97), (97, 1)),
    ((64, 4096), (4096, 1)),
    ((97, 33), (1, 97)),
    ((129, 2048), (1, 129)),
    # 3D contiguous / transposed
    ((5, 7, 11), (77, 11, 1)),
    ((97, 131, 5), (655, 5, 1)),
    ((7, 5, 11), (11, 77, 1)),
    ((131, 97, 5), (5, 655, 1)),
]
HELDOUT_IRREGULAR_SHAPE_STRIDES = [((8, 8, 8, 8, 8), (1, 4096, 17, 257, 999))]
HELDOUT_UPSAMPLE_SHAPES = [
    (7, 3, 65, 129),
    (2, 5, 201, 203),
    (1, 1, 255, 257),
]
HELDOUT_KRON_SHAPES = [
    [(), (7,)],
    [(4, 5), (2, 3)],
    [(3, 3, 3), (2, 2, 2)],
]

if HELDOUT_MODE:
    UT_SHAPES_1D = UT_SHAPES_1D + HELDOUT_UT_SHAPES_1D
    UT_SHAPES_2D = UT_SHAPES_2D + HELDOUT_UT_SHAPES_2D
    POINTWISE_SHAPES = POINTWISE_SHAPES + HELDOUT_POINTWISE_SHAPES
    SPECIAL_SHAPES = SPECIAL_SHAPES + HELDOUT_SPECIAL_SHAPES
    DISTRIBUTION_SHAPES = DISTRIBUTION_SHAPES + HELDOUT_DISTRIBUTION_SHAPES
    REDUCTION_SHAPES = REDUCTION_SHAPES + HELDOUT_REDUCTION_SHAPES
    REDUCTION_SMALL_SHAPES = REDUCTION_SMALL_SHAPES + HELDOUT_REDUCTION_SMALL_SHAPES
    STACK_SHAPES = STACK_SHAPES + HELDOUT_STACK_SHAPES
    SHAPE_STRIDES = SHAPE_STRIDES + HELDOUT_SHAPE_STRIDES
    IRREGULAR_SHAPE_STRIDES = IRREGULAR_SHAPE_STRIDES + HELDOUT_IRREGULAR_SHAPE_STRIDES
    UPSAMPLE_SHAPES = UPSAMPLE_SHAPES + HELDOUT_UPSAMPLE_SHAPES
    KRON_SHAPES = KRON_SHAPES + HELDOUT_KRON_SHAPES


# Add some test cases with zeor-dimensional tensor and zero-sized tensors.
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64] if fp64_is_supported else FLOAT_DTYPES
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64]
BOOL_TYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex32, torch.complex64]

SCALARS = [0.001, -0.999, 100.001, -111.999]
STACK_DIM_LIST = [-2, -1, 0, 1]

ARANGE_START = [0] if TO_CPU else [0, 1, 3]

def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    # if upcast and UPCAST:
    #     if ref_inp.is_complex():
    #         ref_inp = ref_inp.to(torch.complex128)
    #     else:
    #         ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def to_cpu(res, ref):
    if TO_CPU:
        res = res.to("cpu")
        assert ref.device == torch.device("cpu")
    return res


def kernelgenbench_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    res = to_cpu(res, ref)
    assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )


def kernelgenbench_assert_equal(res, ref, equal_nan=False):
    res = to_cpu(res, ref)
    assert_equal(res, ref, equal_nan=equal_nan)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp


def init_seed(seed):
    seed = int(seed) + SEED_OFFSET
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


