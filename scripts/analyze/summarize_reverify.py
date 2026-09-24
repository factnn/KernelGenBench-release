#!/usr/bin/env python3
"""Summarise the re-verification arms into the tables used in the paper.

Inputs produced by scripts/analyze/reverify_corpus.py:
  baseline.json   published protocol (+ per-comparison tolerance audit)
  heldout.json    published shapes + held-out shapes/strides and a fresh seed
  clone_free.json published protocol with the in-timed-region clone removed

Usage:
  python scripts/analyze/summarize_reverify.py \
      --baseline runs/baseline_gpus0123.json \
      --heldout  runs/heldout_gpus4567.json \
      [--clone-free runs/clone_free.json] \
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
    ap.add_argument("--clone-free", dest="clone_free", default=None)
    ap.add_argument("--tolerance-bound", dest="tolerance_bound", default=None)
    ap.add_argument("--timing-clone-cost", dest="timing_clone_cost", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = by_op(load(args.baseline))
    held = by_op(load(args.heldout))
    clon = by_op(load(args.clone_free))
    bound = load(args.tolerance_bound)
    tcost = load(args.timing_clone_cost)

    lines = []
    add = lines.append

    add("# KernelGenBench re-verification of previously accepted kernels\n")
    add("No model is re-invoked: every number below comes from re-running the "
        "released verification pipeline on kernels the pipeline had already "
        "accepted.\n")

    # ---------------------------------------------------------------- M5
    add("## 1. Tolerance sensitivity (M5)\n")
    tol_ops = {op: r["tolerance"] for op, r in base.items()
               if r.get("tolerance")}
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

    # ---------------------------------------------------------------- M7
    if clon:
        add("\n## 3. Clone-free timing sensitivity (M7)\n")
        pairs = []
        for op, r in clon.items():
            b = base.get(op)
            if not b:
                continue
            s_clone = b.get("speedup")
            s_free = r.get("speedup")
            if isinstance(s_clone, (int, float)) and isinstance(s_free, (int, float)) \
                    and s_clone > 0 and s_free > 0:
                pairs.append((op, s_clone, s_free, r.get("passed")))
        add(f"Operators with a comparable speedup in both timing modes: "
            f"**{len(pairs)}**\n")
        if pairs:
            ratio = [f / c for _, c, f, _ in pairs]
            gm = math.exp(sum(math.log(x) for x in ratio) / len(ratio))
            moved = [p for p in pairs if (p[3] is not None and
                                          (p[3] and p[2] > 0))]
            add("| operator | speedup (published) | speedup (clone-free) | ratio |")
            add("|---|---|---|---|")
            for op, c, f, _ in sorted(pairs, key=lambda p: -(p[2] / p[1])):
                add(f"| `{op}` | {fnum(c)}x | {fnum(f)}x | {fnum(f / c)}x |")
            add("")
            add(f"Geometric-mean ratio (clone-free / published): **{fnum(gm)}x**\n")
            below = sum(1 for _, c, _, _ in pairs if c < 1.0)
            below_free = sum(1 for _, _, f, _ in pairs if f < 1.0)
            add(f"- operators measured below 1.0x with the published protocol: "
                f"**{below}/{len(pairs)}**")
            add(f"- operators measured below 1.0x with clone-free timing: "
                f"**{below_free}/{len(pairs)}**\n")

    # ------------------------------------------------- tolerance bound probe
    if bound:
        add("\n## 1b. How much error each tolerance rule admits (probe)\n")
        add("Only eight operators in the 110-operator ATen suite pass a reduction "
            "length as `D_reduce`; every other operator uses the default "
            "`atol = 1e-4`. The probe below perturbs the reference output of the "
            "published test cases and reports the largest relative error that "
            "still passes each rule.\n")
        add("| probe | shape | dtype | D_reduce | published atol | "
            "atol / &#124;ref&#124;max | max rel. error admitted (published) | "
            "(sqrt) | (const) |")
        add("|---|---|---|---|---|---|---|---|---|")
        for p in bound.get("probes", []):
            a = p["admitted_relative_error"]
            add(f"| `{p['probe']}` | {tuple(p['shape'])} | {p['dtype']} | "
                f"{p['reduce_dim']} | {p['published_atol']:.4g} | "
                f"{p['atol_over_output_scale']:.3g} | "
                f"{a['published']:.3g} | {a['sqrt']:.3g} | {a['const']:.3g} |")
        add("")
        ok = [p["probe"] for p in bound.get("probes", [])
              if p.get("order_error_passes_all_rules")]
        if ok:
            add("A *correct* kernel's floating-point reduction-order error still "
                "passes all three rules for: " + ", ".join(f"`{o}`" for o in ok) +
                ". Tightening the rule therefore does not reject legitimate "
                "kernels on these cases.\n")

    # ---------------------------------------------------- clone cost + distortion
    if tcost:
        add("\n## 3b. Why the clone compresses speedups\n")
        add("Fixed overhead that the harness adds to *both* the reference and the "
            "candidate measurement, and the speedup it reports for a kernel that "
            "is genuinely 2x the reference.\n")
        add("| operator | measured (harness) | kernel only | overhead | "
            "overhead share | a true 2.00x is reported as |")
        add("|---|---|---|---|---|---|")
        for c in tcost.get("cases", []):
            two = next((d for d in c["distortion"]
                        if abs(d["true_speedup"] - 2.0) < 1e-9), None)
            add(f"| `{c['op']}` | {c['ms_clone_inclusive']:.4f} ms | "
                f"{c['ms_clone_free']:.4f} ms | {c['effective_overhead']:.4f} ms | "
                f"{100 * c['overhead_share_of_measured_latency']:.1f}% | "
                f"{two['reported_speedup']:.3f}x |" if two else
                f"| `{c['op']}` | {c['ms_clone_inclusive']:.4f} ms | - | - | - | - |")
        add("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
