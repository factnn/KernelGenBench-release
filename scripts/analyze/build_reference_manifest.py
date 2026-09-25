#!/usr/bin/env python3
"""Build the manifest of the shipped reference candidates.

The manifest records, for every candidate in the release directory, the file
hash and the number of test cases it passes under the published suite and under
the held-out suite.  It is generated from the two arm outputs so that the file
cannot drift from what the verifier actually reports.

Usage:
  python scripts/analyze/build_reference_manifest.py \
      --candidates reproducibility/reference_candidates \
      --baseline runs/ref_baseline.json \
      --heldout runs/ref_heldout.json \
      --out reproducibility/reference_candidates/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NOTE = ("Reference candidates for exercising the released verification pipeline "
        "end to end. Each entry passes the published suite and the held-out "
        "suite; the held-out grids add shapes and strides not exposed to the "
        "generator or to the agent.")


def parse_name(path: Path):
    stem = path.name[:-3] if path.name.endswith(".py") else path.name
    if "__" not in stem:
        return None
    namespace, operator = stem.split("__", 1)
    return namespace, operator


def load_results(path):
    doc = json.loads(Path(path).read_text())
    return {r["op"]: r for r in doc.get("results", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_results(args.baseline)
    held = load_results(args.heldout)

    entries = []
    for path in sorted(Path(args.candidates).glob("*.py")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        op = f"{parsed[0]}::{parsed[1]}"
        b = base.get(op, {})
        h = held.get(op, {})
        entries.append({
            "op": op,
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "baseline_tests": b.get("total_tests"),
            "baseline_passed": b.get("passed_tests"),
            "heldout_tests": h.get("total_tests"),
            "heldout_passed": h.get("passed_tests"),
        })

    missing = [e["op"] for e in entries
               if e["baseline_tests"] is None or e["heldout_tests"] is None]
    if missing:
        raise SystemExit(f"no arm output for: {', '.join(missing)}")

    failing = [e["op"] for e in entries
               if e["baseline_passed"] != e["baseline_tests"]
               or e["heldout_passed"] != e["heldout_tests"]]
    if failing:
        raise SystemExit(f"candidate does not pass both suites: {', '.join(failing)}")

    out = {"note": NOTE, "count": len(entries), "entries": entries}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}: {len(entries)} candidates, "
          f"{sum(e['baseline_tests'] for e in entries)} published and "
          f"{sum(e['heldout_tests'] for e in entries)} held-out test cases")


if __name__ == "__main__":
    main()
