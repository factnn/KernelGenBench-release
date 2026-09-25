# Reproducibility material

This directory holds everything the paper's validation appendix needs beyond the
framework itself.

## `reference_candidates/`

Twenty saved candidates (13 ATen, 4 cuBLAS, 3 vLLM) that exercise the released
verification pipeline end to end. Each one passes the published test suite and
the held-out suite, whose grids add shapes and strides that are never exposed to
the generator or to the agent. `manifest.json` records the SHA-256 of every file
together with the number of test cases it passes in both suites.

The solution corpus behind the paper's main tables is **not** released: the
benchmark is also used for vendor kernel-development challenges. The reference
candidates are what a reader can re-verify.

## Re-verifying the candidates

```bash
PY=python                       # an interpreter with torch, triton and the package installed
K=reproducibility/reference_candidates

# 1. Published protocol, recording the tolerance every passing comparison needs.
$PY scripts/analyze/reverify_corpus.py --kernels $K --policy baseline \
    --out runs/ref_baseline.json --audit-dir runs/ref_audit --jobs 8 --gpus 0,1,2,3

# 2. Held-out shapes, strides and input values.
$PY scripts/analyze/reverify_corpus.py --kernels $K --policy heldout \
    --out runs/ref_heldout.json --jobs 8 --gpus 0,1,2,3

```

No model is invoked by either command. The policy switches they set
(`KGB_HELDOUT`, `KGB_SEED_OFFSET`, `KGB_AUDIT`) are described in the top-level
`README.md`.

## What the two arms measure

- **Tolerance.** Re-verifying the 20 candidates with `KGB_AUDIT` set records
  1,153 passing floating-point comparisons from the 9 candidates whose outputs
  are floating-point. The largest absolute tolerance any of them requires is
  1.2e-7, three orders of magnitude below the published `atol`.
- **Held-out inputs.** Every candidate gains test cases in the held-out grids
  rather than a new random draw alone, and all 20 still pass. Counts per
  candidate are in `reference_candidates/manifest.json`, and the exact grids are
  in `src/kernelgenbench/accuracy/`.

## Environment

The measurements above were taken on NVIDIA A100-40GB with the PyTorch, Triton
and CUDA versions listed in `requirements.txt` and the setup steps in the
top-level `README.md`.
