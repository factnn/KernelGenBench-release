#!/usr/bin/env python3
"""Summarise the re-verification arms into the tables used in the paper.

Inputs produced by scripts/analyze/reverify_corpus.py:
  baseline.json   published protocol (+ per-comparison tolerance audit)
  heldout.json    published shapes + held-out shapes/strides and a fresh seed

Usage:
  python scripts/analyze/summarize_reverify.py \
      --baseline runs/baseline_gpus0123.json \
      --heldout  runs/heldout_gpus4567.json \
      --out runs/summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def by_op(doc):
    return {r["op"]: r for r in (doc or {}).get("results", [])}


def source_of(op):
    return op.split("::")[0]


def fnum(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = by_op(load(args.baseline))
    held = by_op(load(args.heldout))

    lines = []
    add = lines.append

    add("# KernelGenBench re-verification of previously accepted kernels\n")
    add("No model is re-invoked: every number below comes from re-running the "
        "released verification pipeline on kernels the pipeline had already "
        "accepted.\n")

    # ---------------------------------------------------------------- M5
    add("## 1. Tolerance sensitivity (M5)\n")
    tol_ops = {op: r["tolerance"] for op, r in base.items()
               if r.get("passed") and r.get("tolerance")}
    n_pass_pub = sum(1 for r in base.values() if r.get("passed"))
    add(f"Operators re-verified under the published rule: **{len(base)}**, "
        f"of which **{n_pass_pub}** pass.\n")
    add(f"Operators with a recorded tolerance audit: **{len(tol_ops)}**.\n")

    if tol_ops:
        const_pass = [op for op, t in tol_ops.items() if t["passes_const"]]
        sqrt_pass = [op for op, t in tol_ops.items() if t["passes_sqrt"]]
        scaled_pass = [op for op, t in tol_ops.items() if t["passes_scaled"]]
        add("| tolerance rule | atol | operators still passing | lost vs published |")
        add("|---|---|---|---|")
        add(f"| published (`1e-4 * D_reduce`) | `1e-4 * D` | "
            f"{len(scaled_pass)} | 0 |")
        add(f"| square-root (`1e-4 * sqrt(D)`) | `1e-4 * sqrt(D)` | "
            f"{len(sqrt_pass)} | {len(scaled_pass) - len(sqrt_pass)} |")
        add(f"| constant (`1e-4`) | `1e-4` | {len(const_pass)} | "
            f"{len(scaled_pass) - len(const_pass)} |")
        add("")

        lost_const = sorted(set(tol_ops) - set(const_pass))
        if lost_const:
            add("Operators that pass the published rule but fail at `atol = 1e-4`:\n")
            add("| operator | reduce_dim | published atol | required atol |")
            add("|---|---|---|---|")
            for op in lost_const:
                t = tol_ops[op]
                add(f"| `{op}` | {t['worst_reduce_dim']} | "
                    f"{fnum(t['published_atol'], 4)} | {fnum(t['max_slack'], 6)} |")
        else:
            add("**No operator that passes the published rule fails at the "
                "constant `atol = 1e-4`.** The reduction-length scaling is "
                "therefore not load-bearing for this corpus.\n")

        worst = max(tol_ops.items(), key=lambda kv: kv[1]["max_slack"])
        add(f"\nLargest required `atol` observed anywhere in the corpus: "
            f"**{fnum(worst[1]['max_slack'], 6)}** on `{worst[0]}` "
            f"(D_reduce = {worst[1]['worst_reduce_dim']}, published atol = "
            f"{fnum(worst[1]['published_atol'], 4)}).\n")
        biggest_d = max(tol_ops.items(), key=lambda kv: kv[1]["max_reduce_dim"])
        add(f"Largest reduction length exercised: "
            f"**{biggest_d[1]['max_reduce_dim']}** on `{biggest_d[0]}`.\n")

    # ---------------------------------------------------------------- M6
    if held:
        add("\n## 2. Held-out shape / stride / value generalization (M6)\n")
        add("Held-out runs use the published grids **plus** shapes and strides "
            "that were never shown to the generator or the agent, and a fresh "
            "input seed.\n")
        common = sorted(set(base) & set(held))
        pub_pass = [op for op in common if base[op].get("passed")]
        both_pass = [op for op in pub_pass if held[op].get("passed")]
        lost = [op for op in pub_pass if not held[op].get("passed")]
        add(f"- operators re-verified in both arms: **{len(common)}**")
        add(f"- passing the published suite: **{len(pub_pass)}**")
        add(f"- of those, still passing on held-out inputs: **{len(both_pass)}** "
            f"({100.0 * len(both_pass) / max(len(pub_pass), 1):.1f}%)")
        add(f"- **lost on held-out inputs: {len(lost)}**\n")
        if lost:
            add("| operator | published | held-out | first held-out error |")
            add("|---|---|---|---|")
            for op in lost:
                err = (held[op].get("error") or "").strip().splitlines()
                err = err[-1][:110] if err else ""
                add(f"| `{op}` | {base[op].get('passed_tests')}/"
                    f"{base[op].get('total_tests')} | "
                    f"{held[op].get('passed_tests')}/"
                    f"{held[op].get('total_tests')} | {err} |")
            per_source = {}
            for op in lost:
                per_source.setdefault(source_of(op), []).append(op)
            add("")
            add("Losses by source: " + ", ".join(
                f"{k} {len(v)}" for k, v in sorted(per_source.items())) + "\n")
        else:
            add("**Every kernel that passes the published suite also passes the "
                "held-out suite.**\n")
