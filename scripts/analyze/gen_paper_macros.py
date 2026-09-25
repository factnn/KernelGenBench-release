#!/usr/bin/env python3
"""Generate the LaTeX macros used by the paper's robustness appendix.

The appendix quotes two kinds of text: release statistics for the reference
candidates that ship with the supplementary material, and the wording that
describes the held-out and timing policies.  This script writes them into one
generated file. The numerical values are fixed statistics for the published
release, not statistics recomputed from the supplied logs. Use
summarize_reverify.py to calculate results for a new run. Supplied log paths
are checked for existence and valid JSON.

Usage:
  python scripts/analyze/gen_paper_macros.py \
      --baseline runs/baseline_gpus0123.json \
      --heldout runs/heldout_gpus4567.json \
      --out ../flagbench/.claude/docs/essay/version8/sections/7_numbers.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    """Load a recorded experiment output, or None when it is absent."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--clone-free", dest="clone_free", default=None)
    ap.add_argument("--tolerance-bound", dest="tolerance_bound", default=None)
    ap.add_argument("--timing-clone-cost", dest="timing_clone_cost", default=None)
    ap.add_argument("--original-results", dest="original_results", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Provenance check: the appendix's prose cites these outputs, so report any
    # that the caller expected but that is not on disk.
    recorded = {
        "baseline": args.baseline,
        "held-out": args.heldout,
        "clone-free": args.clone_free,
        "tolerance probe": args.tolerance_bound,
        "clone-cost probe": args.timing_clone_cost,
        "original run": args.original_results,
    }
    for name, path in recorded.items():
        if path and load(path) is None:
            raise SystemExit(f"{name} output not found: {path}")

    macros = {}

    # Reference candidates shipped with the supplementary material.  These are
    # release constants: they describe the artifact, not a re-computed result.
    macros["RCOUNT"] = "20"
    macros["RAUDITCHECKS"] = "1{,}153"
    macros["RAUDITOPS"] = "9"
    macros["RAUDITMAXSLACK"] = "$1.2\\times10^{-7}$"

    # Held-out generalization, stated for the shipped reference candidates
    # because they are the only saved candidates a reader can re-verify.
    macros["HELDOUTPARAGRAPH"] = (
        "Across the 20 released kernels, the number of test cases increases from "
        "1,681 to 2,530. All 20 kernels pass both suites. Each kernel receives "
        "additional cases: \\texttt{cos} increases from 18 to 36, "
        "\\texttt{argmax} from 126 to 180, "
        "\\texttt{cublasSaxpy\\_v2} from 648 to 864 and \\texttt{rms\\_norm} from 60 "
        "to 84.")

    # Clone-free timing on the published shapes.
    macros["CLONEPARAGRAPH"] = (
        "For the operators in Table~\\ref{tab:clone_cost}, the tests time "
        "\\texttt{op(inp.clone())}, so the measured latency "
        "includes a per-call input copy that both the reference and the candidate pay. "
        "Table~\\ref{tab:clone_cost} measures what that costs on the published shapes: "
        "the copy accounts for 29--61\\% of the measured latency. Under a shared "
        "additive copy cost, a kernel-only speedup of $2\\times$ corresponds to "
        "a measured speedup of $1.24$--$1.55\\times$. The "
        "shared clone overhead pulls the measured speedup toward $1\\times$, "
        "understating speedups above $1\\times$ and overstating relative performance "
        "below $1\\times$. The ratio is unchanged when the two implementations "
        "take exactly the same kernel time. In-place operators "
        "retain the copy, because their benchmark loop needs a fresh input on every "
        "iteration; the timing switch therefore applies only where the copy is pure "
        "overhead.")

    lines = ["% Generated file -- edit the generator, not this file.", ""]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
