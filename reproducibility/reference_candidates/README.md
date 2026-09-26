# Reference candidates

A small set of saved candidates that exercise the released verification pipeline
end to end. They are generation outputs from the Claude~Code corpus behind the
paper's tables, released as reference material for the pipeline rather than as
hand-written reference implementations. Every entry passes both the published test suite and the held-out
suite, whose grids add shapes and strides that were never exposed to the
generator or to the agent. Use them to check that the infrastructure mounts a
candidate, runs the operator's tests, applies the anti-hack checks and reports
performance as expected:

```bash
python agent_bench/tools/verify_single.py \
    --code reproducibility/reference_candidates/aten__cos.py \
    --op cos --namespace aten --output-json
```

`manifest.json` records the SHA-256 of every file together with the number of
test cases each candidate passes in the published and held-out suites.

These are reference candidates for validating the pipeline, not the solution
corpus behind the paper's tables: the full set of saved candidates is not
released, because the benchmark is also used for vendor kernel-development
challenges.
