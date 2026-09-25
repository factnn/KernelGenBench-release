# Frozen replay study

This directory contains a frozen code snapshot, 152 saved candidates, a hash manifest, an environment record, and a predeclared 21-task repeat subset. No model is invoked.

The initial pre-dispatch-fix results are archived separately and **invalid as candidate verification**: the namespace classifier failed to recognize `aten::bmm` as ATen, causing the candidate to be skipped. A deliberately zero-output candidate was accepted. The corrected classifier and an empty-registration guard are in the new snapshot. The negative control must fail before the full replay is accepted.

Run in an environment matching `environment.json`:

```bash
python run_study.py --python /path/to/python --gpus 0,1,2,3 --out /new/output/directory
python summarize_study.py --results /new/output/directory
```

Use a new output directory on every run. The study verifies manifest hashes before execution. GPUs must be reserved for this run; within one driver, each GPU handles at most one operator at a time. Baseline and held-out arms cover all saved candidates; two additional baseline repeats and a clone-free arm cover the predeclared subset. Functional outcomes are distinct from clean leaderboard results: runtime L2/L3 anti-hack checks are not enabled by this entry point. Performance noise and model-generation randomness are separate questions.

The environment differs from an unspecified historical suite snapshot, so these results cannot certify the original main tables. The original API model trajectories are not rerun.

## Scope of this release

The saved solution corpus used for the study is **not** released: the benchmark is
also used for vendor kernel-development challenges, so publishing accepted
solutions would compromise future evaluations. `manifest.json` is the hash record
of the corpus that was studied, and `run_study.py` documents the protocol; it
needs that corpus to run. To exercise the released pipeline end to end, use the
reference candidates in `../reference_candidates/`, which pass both the published
and the held-out suites.
