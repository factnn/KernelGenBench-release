#!/usr/bin/env python3
"""Measure how large an error each tolerance rule admits, on the real test grid.

Only eight operators in the 110-operator ATen suite pass a reduction length as
``D_reduce`` (``sum``, ``mm``, ``matmul``, ``linear``, ``baddbmm``,
``baddbmm_backward``, ``cumsum``, ``_softmax_backward``); every other operator
uses the default ``atol = 1e-4``.  The released re-verification corpus does not
contain saved kernels for most of those eight, so this probe measures the bound
directly, on the exact shapes and dtypes of the published suite:

  * the largest relative perturbation of the reference output that still passes
    under each rule (published / sqrt / constant), and
  * the error a *correct* kernel incurs from floating-point reduction order
    (float64 evaluation cast back), which is what a tolerance rule must
    accommodate without failing legitimate kernels.

Usage:
  python scripts/analyze/tolerance_bound_probe.py --out runs/tolerance_bound.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sandbox.utils.accuracy_utils import RESOLUTION  # noqa: E402

# label, shape, dtype, D_reduce kind, builder
#   kind: "numel"  -> D = numel(tensor)   (test_accuracy_sum_without_dim)
#         "K"      -> D = K of the GEMM    (mm / matmul / linear / baddbmm)
#         "dim0"   -> D = shape[0]         (cumsum with dim=0)
#         "dim1"   -> D = shape[1]         (cumsum with dim=1)
#         "one"    -> D = 1, published default (softmax, mean, ...)
PROBES = [
    ("sum_f32",            (200, 40999, 3), torch.float32,  "numel", "sum"),
    ("sum_f16",            (200, 40999, 3), torch.float16,  "numel", "sum"),
    ("sum_bf16",           (200, 40999, 3), torch.bfloat16, "numel", "sum"),
    ("mm_M1024_K1024",     (1024, 1024),    torch.float32,  "K",     "mm"),
    ("mm_M128_K256",       (128, 256),      torch.float32,  "K",     "mm"),
    ("mm_M4096_K5120_f16", (4096, 5120),    torch.float16,  "K",     "mm"),
    ("cumsum_dim0",        (200, 40999, 3), torch.float32,  "dim0",  "cumsum"),
    ("cumsum_dim1",        (200, 40999, 3), torch.float32,  "dim1",  "cumsum"),
    ("mean_default",       (200, 40999, 3), torch.float32,  "one",   "mean"),
    ("softmax_default",    (4096, 256),     torch.float32,  "one",   "softmax"),
]

EPSILONS = [10 ** (-7 + 0.125 * i) for i in range(57)]  # 1e-7 .. 1e0


def reduce_dim_value(kind: str, shape) -> int:
    if kind == "numel":
        n = 1
        for s in shape:
            n *= s
        return n
    if kind == "K":
        return int(shape[1])
    if kind == "dim0":
        return int(shape[0])
    if kind == "dim1":
        return int(shape[1])
    if kind == "one":
        return 1
    raise ValueError(kind)


def required_atol(cand, ref, rtol) -> float:
    slack = (cand - ref).abs() - rtol * ref.abs()
    return float(slack.max())


def passes(required: float, rule: str, d: int) -> bool:
    if rule == "published":
        return required <= 1e-4 * max(d, 1)
    if rule == "sqrt":
        return required <= 1e-4 * math.sqrt(max(d, 1))
    if rule == "const":
        return required <= 1e-4
    raise ValueError(rule)


def build(builder, shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    if builder == "sum":
        return torch.sum(x), torch.sum(x.double()).to(dtype)
    if builder == "mean":
        return torch.mean(x), torch.mean(x.double()).to(dtype)
    if builder == "mm":
        m, k = shape
        a = torch.randn((m, k), dtype=dtype, device=device)
        b = torch.randn((k, k), dtype=dtype, device=device)
        return torch.mm(a, b), None
    if builder == "cumsum":
        return torch.cumsum(x, dim=0), None
    if builder == "softmax":
        return torch.softmax(x, dim=0), None
    raise ValueError(builder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(0)
    rows = []
    for label, shape, dtype, kind, builder in PROBES:
        d = reduce_dim_value(kind, shape)
        ref, order_cand = build(builder, shape, dtype, args.device)
        rtol = RESOLUTION[dtype]
        published_atol = 1e-4 * max(d, 1)
        scale = float(ref.abs().max())

        sweep = []
        for eps in EPSILONS:
            cand = (ref * (1.0 + eps)).to(dtype)
            req = required_atol(cand, ref, rtol)
            sweep.append({
                "eps": eps,
                "required_atol": req,
                "pass_published": passes(req, "published", d),
                "pass_sqrt": passes(req, "sqrt", d),
                "pass_const": passes(req, "const", d),
            })
        admitted = {r: max((s["eps"] for s in sweep if s[f"pass_{r}"]), default=0.0)
                    for r in ("published", "sqrt", "const")}

        order_req = required_atol(order_cand, ref, rtol) if order_cand is not None else None
        rows.append({
            "probe": label,
            "shape": list(shape),
            "dtype": str(dtype).replace("torch.", ""),
            "reduce_dim": d,
            "rtol": rtol,
            "published_atol": published_atol,
            "atol_over_output_scale": published_atol / max(scale, 1e-30),
            "ref_abs_max": scale,
            "admitted_relative_error": admitted,
            "order_error_required_atol": order_req,
            "order_error_passes_all_rules": (
                None if order_req is None else
                all(passes(order_req, r, d) for r in ("published", "sqrt", "const"))),
            "sweep": sweep,
        })
        print(f"{label:20s} D={d:<9d} atol={published_atol:<9.4g} "
              f"atol/|ref|max={published_atol / max(scale, 1e-30):<9.4g} "
              f"admitted: pub={admitted['published']:<9.4g} "
              f"sqrt={admitted['sqrt']:<9.4g} const={admitted['const']:<9.4g}"
              + (f"  order_err_atol={order_req:.3g}" if order_req is not None else ""))
        del ref
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"probes": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
