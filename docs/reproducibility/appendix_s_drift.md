# Appendix S: recorded outcome drift

This is a read-only reconciliation of saved records, not a new GPU evaluation. Of 145 shared operators, 129 agree, 11 change from failed to passed, and 5 change from passed to failed. Historical and current records need not share test cases, environment, or validation implementation. Candidate hashes identify files currently on disk; they do not prove those bytes were used in the historical run.

| Operator | Original passed cases | Current passed cases | Recorded evidence |
|---|---:|---:|---|
| `aten::affine_grid_generator` | 5/36 | 72/72 | Historical traceback reaches numerical comparison; case count changes from 36 to 72. Cause unresolved. |
| `aten::bernoulli` | 0/9 | 90/90 | Historical traceback reaches numerical comparison; case count changes from 9 to 90. Cause unresolved. |
| `aten::binary_cross_entropy_with_logits` | 0/0 | 216/216 | Historical traceback is in source checking before tests; message is truncated. Cause unresolved. |
| `aten::bmm` | 9/9 | 0/9 | Current test raises NameError: shape is undefined. The test body also calls cat rather than bmm. |
| `aten::div_` | 96/96 | 90/96 | Current truncation test raises NameError: inp1 is undefined. |
| `aten::index` | 30/30 | 28/29 | Current empty-indices reference call raises an internal PyTorch assertion. Root cause unresolved. |
| `aten::index_put_` | 44/44 | 33/43 | Current accumulation test raises UnboundLocalError for inp before assignment. |
| `aten::pow` | 162/162 | 90/162 | Current test accesses imag on a non-complex tensor and raises RuntimeError. |
| `aten::sort` | 0/0 | 320/320 | Historical run timed out after 600 seconds; current replay passes. Cause unresolved. |
| `cublas::cublasCgemvStridedBatched` | 0/0 | 432/432 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasCgemv_v2` | 0/0 | 1152/1152 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasDgemmStridedBatched` | 0/0 | 864/864 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasDgemvStridedBatched` | 0/0 | 432/432 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasSaxpy_v2` | 0/0 | 648/648 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasSscal_v2` | 0/0 | 216/216 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |
| `cublas::cublasZgemmStridedBatched` | 0/0 | 768/768 | Historical process exited with code -6 before test results; current replay passes. Exit code alone does not establish cause. |

The five pass-to-fail records expose test/reference-stage exceptions; they are not evidence of five numerically incorrect generated kernels. The source-checking traceback, numerical-comparison traces, timeout, and process exits in the eleven reverse transitions do not by themselves identify the cause of improved outcomes. Fixing tests would create another suite version and requires paired re-verification before replacing the appendix counts.

The JSON companion records input-result hashes and candidate hashes without local absolute paths. Full historical test snapshots and runtime environments are not reconstructed by this report.
