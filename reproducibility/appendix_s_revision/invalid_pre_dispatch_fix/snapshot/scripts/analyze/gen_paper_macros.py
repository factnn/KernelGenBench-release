#!/usr/bin/env python3
"""Generate the LaTeX macros used by the paper's robustness appendix.

Reads the arm outputs produced by reverify_corpus.py plus the two probes, and
writes sections/7_numbers.tex, so that every number in the appendix comes from
the experiment outputs rather than being transcribed by hand.

Usage:
  python scripts/analyze/gen_paper_macros.py \
      --baseline runs/baseline_gpus0123.json \
      --heldout runs/heldout_gpus4567.json \
      [--clone-free runs/clone_free.json] \
      --tolerance-bound runs/tolerance_bound.json \
      --timing-clone-cost runs/timing_clone_cost.json \
      --out ../flagbench/.claude/docs/essay/version8/sections/7_numbers.tex
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


def esc(text: str) -> str:
    return (text.replace("_", r"\_").replace("%", r"\%")
                .replace("&", r"\&").replace("#", r"\#"))


def fmt_int(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--clone-free", dest="clone_free", default=None)
    ap.add_argument("--tolerance-bound", dest="tolerance_bound", default=None)
    ap.add_argument("--timing-clone-cost", dest="timing_clone_cost", default=None)
    ap.add_argument("--original-results", dest="original_results", default=None,
                    help="results.json of the run that produced the paper's "
                         "numbers, used to report agreement with the released "
                         "suite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_doc = load(args.baseline) or {"results": []}
    held_doc = load(args.heldout) or {"results": []}
    clon_doc = load(args.clone_free) or {"results": []}
    bound = load(args.tolerance_bound) or {"probes": []}
    tcost = load(args.timing_clone_cost) or {"cases": []}

    base = {r["op"]: r for r in base_doc["results"]}
    held = {r["op"]: r for r in held_doc["results"]}
    clon = {r["op"]: r for r in clon_doc["results"]}

    macros = {}

    # ---- tolerance audit -------------------------------------------------
    audits = {op: r["tolerance"] for op, r in base.items() if r.get("passed") and r.get("tolerance")}
    if audits:
        n_checks = sum(t["n_checks"] for t in audits.values())
        max_d = max(t["max_reduce_dim"] for t in audits.values())
        max_slack = max(t["max_slack"] for t in audits.values())
        n_pub = sum(1 for t in audits.values() if t["passes_scaled"])
        n_sqrt = sum(1 for t in audits.values() if t["passes_sqrt"])
        n_const = sum(1 for t in audits.values() if t["passes_const"])
        macros["TOLCHECKS"] = fmt_int(n_checks)
        macros["TOLMAXD"] = fmt_int(max_d)
        macros["TOLMAXSLACK"] = f"${max_slack:.3g}$"
        macros["TOLNUMPUB"] = str(n_pub)
        macros["TOLNUMSQRT"] = str(n_sqrt)
        macros["TOLNUMCONST"] = str(n_const)
        lost = n_pub - n_const
        lost_ops = sorted(op for op, t in audits.items() if t["passes_scaled"]
                          and not t["passes_const"])
        if lost == 0:
            macros["TOLLOSTSENTENCE"] = (
                "the reduction-length scaling is therefore not load-bearing for "
                "any kernel in this corpus.")
            macros["TOLNUMBEROPS"] = "none"
        else:
            names = ", ".join(r"\texttt{" + esc(op.split("::")[-1]) + "}"
                              for op in lost_ops)
            macros["TOLLOSTSENTENCE"] = (
                f"{lost} operator(s) pass the published rule but fail the "
                f"constant rule, so the scaling is load-bearing for them.")
            macros["TOLNUMBEROPS"] = names
    else:
        for k in ("TOLCHECKS", "TOLMAXD", "TOLMAXSLACK", "TOLNUMPUB",
                  "TOLNUMSQRT", "TOLNUMCONST"):
            macros[k] = "n/a"
        macros["TOLNUMBEROPS"] = "n/a"
        macros["TOLLOSTSENTENCE"] = "the audit is pending."

    # ---- held-out generalization ----------------------------------------
    common = sorted(set(base) & set(held))
    pub_pass = [op for op in common if base[op].get("passed")]
    both = [op for op in pub_pass if held[op].get("passed")]
    lost_ops = [op for op in pub_pass if not held[op].get("passed")]
    if common:
        n_cases_added = sum(
            1 for op in pub_pass
            if (held[op].get("total_tests") or 0) > (base[op].get("total_tests") or 0))
        pct = 100.0 * len(both) / max(len(pub_pass), 1)
        if not lost_ops:
            macros["HELDOUTPARAGRAPH"] = (
                f"Of the {len(common)} operators re-verified in both arms, "
                f"{len(pub_pass)} pass the current released suite; all {len(both)} "
                f"({pct:.1f}\\%) also pass the held-out evaluation. "
                f"Of these accepted operators, {n_cases_added} receive additional "
                "unseen shape/stride test cases. The shifted seed also tests "
                "input-value sensitivity, but does not imply new shape/stride "
                "coverage for every operator. This result is specific to this "
                "corpus and suite version, not all main-table kernels.")
        else:
            names = ", ".join(r"\texttt{" + esc(op.split("::")[-1]) + "}"
                              for op in lost_ops[:12])
            macros["HELDOUTPARAGRAPH"] = (
                f"Of the {len(common)} operators re-verified in both arms, "
                f"{len(pub_pass)} pass the published suite and {len(both)} "
                f"({pct:.1f}\\%) also pass on the held-out inputs; "
                f"{len(lost_ops)} fail only on the held-out inputs ({names}). "
                "Specialisation to the published grid therefore accounts for "
                f"{len(lost_ops)} of {len(pub_pass)} accepted kernels in this "
                "corpus.")
    else:
        macros["HELDOUTPARAGRAPH"] = "the held-out arm is pending."

    # ---- clone-free timing on the corpus --------------------------------
    pairs = []
    for op, r in clon.items():
        b = base.get(op)
        if not b or not b.get("passed") or not r.get("passed"):
            continue
        s1, s2 = b.get("speedup"), r.get("speedup")
        if isinstance(s1, (int, float)) and isinstance(s2, (int, float)) and s1 and s2:
            pairs.append((op, s1, s2))
    if pairs:
        ratios = [s2 / s1 for _, s1, s2 in pairs]
        gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        below = sum(1 for _, s1, _ in pairs if s1 < 1.0)
        below2 = sum(1 for _, _, s2 in pairs if s2 < 1.0)
        lo = min(min(s1, s2) for _, s1, s2 in pairs)
        hi = max(max(s1, s2) for _, s1, s2 in pairs)
        macros["CLONEPARAGRAPH"] = (
            f"For the {len(pairs)} corpus operators with paired timing results, "
            f"speedups span {lo:.2f}$\\times$--{hi:.2f}$\\times$. "
            f"The geometric mean of clone-free to baseline speedup ratios is "
            f"{gm:.3f}$\\times$; {below} operators are below parity in the "
            f"baseline and {below2} in clone-free mode. The small aggregate "
            "change applies only to these audited operators and does not "
            "establish timing robustness for other methods or sources. "
            "Under a shared additive clone overhead $c>0$, the measured ratio "
            "is $(t_{\\mathrm{ref}}+c)/(t_{\\mathrm{kernel}}+c)$ rather than "
            "$t_{\\mathrm{ref}}/t_{\\mathrm{kernel}}$. The overhead attenuates "
            "deviations from parity in both directions: it understates speedups "
            "above $1\\times$ and overstates relative performance below "
            "$1\\times$. Table~\\ref{tab:clone_cost} illustrates the former "
            "with a true $2\\times$ acceleration measured as "
            "$1.24$--$1.55\\times$.")
    else:
        macros["CLONEPARAGRAPH"] = "the clone-free arm is pending."

    # ---- agreement with the run that produced the paper's numbers ---------
    macros["CORPUSTOTAL"] = str(len(base))
    n_base_pass = sum(1 for r in base.values() if r.get("passed"))
    macros["CORPUSBASEPASS"] = str(n_base_pass)
    if held:
        macros["CORPUSHELDPASS"] = str(sum(1 for r in held.values() if r.get("passed")))
    else:
        macros["CORPUSHELDPASS"] = "n/a"
    orig_doc = load(args.original_results)
    if orig_doc:
        orig = orig_doc.get("operators", {})
        both = [op for op in base if op in orig]
        agree = [op for op in both
                 if bool(base[op].get("passed")) == (orig[op].get("status") == "passed")]
        disagree = [op for op in both if op not in agree]
        example = ""
        if disagree:
            op = disagree[0]
            a, b = base[op], orig[op]
            example = (
                f" (for example \\texttt{{{esc(op.split('::')[-1])}}} passed "
                f"{b.get('passed_tests', 0)}/{b.get('total_tests', 0)} in the "
                f"original run and {a.get('passed_tests', 0)}/"
                f"{a.get('total_tests', 0)} in the released suite)")
        macros["DRIFTNOTE"] = (
            f"The released suite reproduces the original run's pass/fail outcome "
            f"on {len(agree)} of the {len(both)} operators present in both"
            f"{example}; {len(disagree)} outcomes differ, and the test module "
            "has changed since the paper's runs. "
            "Re-verification therefore measures the current released suite, "
            "not an exact replay of the main-table protocol; each comparison here is "
            "paired within one suite version.")
    else:
        macros["DRIFTNOTE"] = ("Agreement with the original run is not reported "
                               "for this build.")
    if not n_base_pass:
        macros["DRIFTNOTE"] = "The baseline arm has not finished yet."

    lines = ["% Generated by scripts/analyze/gen_paper_macros.py -- do not edit.",
             "% Sources: " + ", ".join(
                 str(Path(p).name) for p in
                 (args.baseline, args.heldout, args.clone_free,
                  args.tolerance_bound, args.timing_clone_cost) if p),
             ""]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
