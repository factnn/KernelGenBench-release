#!/usr/bin/env python3
"""Quantify how much the in-timed-region clone distorts reported speedups.

Most parameterised tests time ``op(inp.clone())``.  If the reference and the
candidate take ``t_ref`` and ``t_cand`` and the clone costs ``c``, the harness
reports ``(t_ref + c) / (t_cand + c)`` instead of ``t_ref / t_cand``.  This
script measures ``c`` and ``t_ref`` for the published shapes of representative
operators and reports

  * the clone's share of the measured latency, and
  * the speedup the harness would report for a kernel that is ``k`` times the
    reference, versus the ``k`` it actually is.

Usage:
  python scripts/analyze/timing_clone_cost.py --out runs/timing_clone_cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton.testing as tt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

CASES = [
    # (name, shape, dtype, fn, inplace); the published tests of every entry
    # put a clone inside the timed region.  In-place operators need that clone
    # to restore their input, so only the non-in-place rows quantify avoidable
    # overhead.
    ("sum",         (200, 40999, 3),  torch.float32, lambda x: torch.sum(x), False),
    ("sum_f16",     (200, 40999, 3),  torch.float16, lambda x: torch.sum(x), False),
    ("mean",        (4096, 256),      torch.float32, lambda x: torch.mean(x), False),
    ("mul",         (1024, 1024),     torch.float32, lambda x: torch.mul(x, 2.0), False),
    ("silu",        (1024, 1024),     torch.float32, lambda x: torch.nn.functional.silu(x), False),
    ("neg",         (1024, 1024),     torch.float32, lambda x: torch.neg(x), False),
    ("rsqrt",       (1024, 1024),     torch.float32, lambda x: torch.rsqrt(x), False),
    ("div_",        (1024, 1024),     torch.float32, lambda x: x.div_(2.0), True),
    ("masked_fill_", (1024, 1024),    torch.float32,
     lambda x: x.masked_fill_(x > 0, 0.0), True),
]

K_FACTORS = [0.5, 0.8, 1.0, 1.25, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rep", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(0)
    rows = []
    for name, shape, dtype, fn, inplace in CASES:
        x = torch.randn(shape, dtype=dtype, device=args.device)
        # clone-inclusive: what the released harness times
        t_with = tt.do_bench(lambda: fn(x.clone()), rep=args.rep)
        # cost of the clone itself, measured *without* any patching
        t_clone = tt.do_bench(lambda: x.clone(), rep=args.rep)
        # clone-free: the kernel alone (identity clone during timing)
        real_clone = torch.Tensor.clone
        torch.Tensor.clone = lambda self, *a, **k: self
        try:
            t_without = tt.do_bench(lambda: fn(x), rep=args.rep)
        finally:
            torch.Tensor.clone = real_clone

        # effective fixed overhead the harness adds to every candidate and to
        # the reference (clone, allocation and bookkeeping)
        effective_overhead = t_with - t_without

        distortion = []
        for k in K_FACTORS:
            reported = (t_without + effective_overhead) / (t_without / k + effective_overhead)
            distortion.append({"true_speedup": k, "reported_speedup": reported})

        rows.append({
            "op": name,
            "inplace": inplace,
            "shape": list(shape),
            "dtype": str(dtype).replace("torch.", ""),
            "ms_clone_inclusive": t_with,
            "ms_clone_free": t_without,
            "ms_clone_only": t_clone,
            "effective_overhead": effective_overhead,
            "overhead_share_of_measured_latency": effective_overhead / t_with,
            "distortion_meaningful": not inplace,
            "distortion": distortion,
        })
        print(f"{name:12s} measured={t_with:8.4f} ms  kernel_only={t_without:8.4f} ms  "
              f"clone_alone={t_clone:8.4f} ms  overhead={effective_overhead:8.4f} ms "
              f"({effective_overhead / t_with:5.1%} of measured)")
        for d in distortion:
            print(f"    true {d['true_speedup']:.2f}x -> reported "
                  f"{d['reported_speedup']:.3f}x")
        del x
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
