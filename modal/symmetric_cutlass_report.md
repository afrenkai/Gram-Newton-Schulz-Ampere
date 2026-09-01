# SM80 symmetric CUTLASS Newton–Schulz benchmark

## Final result

Turing Slurm job `2241410` completed successfully on `NVIDIA A100-SXM4-80GB`.
All numbers below are warmed steady-state medians from 5 warmups and 20 CUDA-event samples.
The input shape is `B x 16384 x 2048`, with BF16 seed 67, `YOU_COEFFICIENTS`,
`eps=1e-7`, and reset iteration `[2]`.

| B | Upstream Standard eager (ms) | Torch GNS (ms) | Triangular CUTLASS GNS (ms) | Standard / CUTLASS | Torch GNS / CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.096 | 4.066 | 3.626 | 1.957x | 1.121x |
| 8 | 52.709 | 30.689 | 24.418 | 2.159x | 1.257x |
| 32 | 224.931 | 130.097 | 96.086 | 2.341x | 1.354x |

The strict baseline is the literal upstream eager implementation at
`e45d0aca7083cb275c9a303220c05c4abecd9187`, with kernels disabled and no compile kwargs.

## Primitive and core timings

| Operation | B | Torch (ms) | Triangular CUTLASS (ms) | Speedup |
|---|---:|---:|---:|---:|
| symmetric square | 1 | 0.093 | 0.083 | 1.123x |
| symmetric square | 8 | 0.611 | 0.404 | 1.511x |
| symmetric square | 32 | 2.535 | 1.512 | 1.676x |
| X.T @ X | 1 | 0.583 | 0.457 | 1.276x |
| X.T @ X | 8 | 3.986 | 2.393 | 1.666x |
| X.T @ X | 32 | 18.741 | 9.037 | 2.074x |
| GNS core | 1 | 3.639 | 2.918 | 1.247x |
| GNS core | 8 | 25.195 | 19.050 | 1.323x |
| GNS core | 32 | 108.170 | 73.849 | 1.465x |

## Cleanup regression

Job `2241292` measured the pre-cleanup source. Job `2241410` measured the final cleaned source
on the same Turing node and software stack, but in a separate run.

| B | Pre-cleanup CUTLASS GNS (ms) | Final CUTLASS GNS (ms) | Change | Pre-cleanup speedup | Final speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.621 | 3.626 | +0.14% | 1.958x | 1.957x |
| 8 | 24.390 | 24.418 | +0.12% | 2.160x | 2.159x |
| 32 | 95.890 | 96.086 | +0.20% | 2.342x | 2.341x |

The cleanup reduced the implementation by 76 lines (`434 -> 383` C++; `315 -> 290` Python).
The isolated B1 core is `2.918 ms`, versus `2.915 ms` before cleanup (`+0.11%`).
The temporary generator-expression version in job `2241361` measured `3.185 ms`; direct checks
removed that host-dispatch regression. Final end-to-end changes are within `0.21%` of job `2241292`.

## Accuracy

- Every end-to-end output is finite.
- Triangular GNS relative error versus Torch GNS is `7.32e-4` to `1.25e-3`.
- Output RMS is `0.007832` for every final batch.
- Orthogonality residual is `0.035409` to `0.035439`, versus `0.035440` to `0.035534` for Standard NS.
- Every promised symmetric primitive and core output has maximum asymmetry exactly `0.0`.

## Generated code

The final extension is `gram_newton_schulz_ampere_sm80_317455dec2ad`; the suffix matches the
first 12 hex digits of the final C++ SHA-256. Removing the unused CC specialization reduced the
shared object from `907,792` to `802,192` bytes.

The final cubin contains six symmetric CUTLASS variants (RR, CR, and RC for FP16/BF16) plus two
mirror variants. Each CUTLASS variant contains 64 HMMA and 24 LDGSTS instructions. CUTLASS register
counts are 240, 252, or 254 depending on layout. Mirror kernels use 32 registers and 2,112 bytes of
shared memory.

Exact PTX, SASS, and resource dumps are stored as:

- `modal/symmetric-cutlass-2241410.ptx`
- `modal/symmetric-cutlass-2241410.sass`
- `modal/symmetric-cutlass-2241410.resources.txt`
- `modal/symmetric-cutlass-2241410.elf-list.txt`

## Algorithmic work

Per matrix, Standard NS performs 15 GEMMs and approximately `1.460289 TFLOP`.
Default-reset GNS performs 18 GEMMs and approximately `0.790274 TFLOP`.
The arithmetic-work ratio is `1.847826x`. The final `1.96x` to `2.34x` measured speedup also reflects
more efficient symmetric execution, not only fewer FLOPs.

## Provenance and cross-run context

- Baseline implementation commit: `fbf6f5d`.
- Final C++ SHA-256: `317455dec2adc1274ff766c1d581be6840e4ad3bcc6b56c0eff386a305eaaec1`.
- Final Python wrapper SHA-256: `6380ef116960632c57688a14f0af5d182daa5a9eef688f14ace4d29feaacb1ad`.
- Turing: A100-SXM4-80GB, driver 595.71.05, Torch 2.11.0+cu130, CUDA runtime 13.0.
- Historical Modal A100 measured the older fastest GNS backend at `1.583x`, `1.729x`, and `1.765x`
  for B1/B8/B32. This is context only because the code, process, and machine run differ.
- Historical Modal H100 Quack measured `2.053x`, `2.530x`, and `2.565x` for B1/B8/B32.
  It is a within-H100 comparison and must not be compared as raw latency against A100.
- The current Turing run did not sample peak memory. The historical Modal B32 runs stayed below
  roughly 25 GiB reserved memory, which is cross-run context only.

Raw final data: `modal/symmetric-cutlass-results-2241410.json` and
`modal/symmetric-cutlass-summary-2241410.csv`.

## Refactor branch regression

Cleanup branch `refactor/ampere-symmetric-cutlass-cleanup` was measured by Turing
job `2241582` against baseline commit `fbf6f5d` / job `2241410`.

| B | Baseline CUTLASS GNS (ms) | Cleanup CUTLASS GNS (ms) | Change | Cleanup Standard / CUTLASS |
|---:|---:|---:|---:|---:|
| 1 | 3.626 | 3.635 | +0.24% | 1.952x |
| 8 | 24.418 | 24.414 | -0.02% | 2.158x |
| 32 | 96.086 | 96.037 | -0.05% | 2.340x |

All triangular primitive, core, and end-to-end timing changes stayed within
`0.95%`; core changes stayed within `0.11%`. Finite outputs, RMS,
orthogonality, and exact symmetry passed. The largest relative-error reporting
difference was `3.5e-10`, caused by the memory-bounded batch-wise norm.

The refactored CUDA extension remains `802,192` bytes. Its complete SASS and
resource-usage dumps are byte-identical to baseline job `2241410`, confirming
that the mathematical grid helper and naming cleanup did not alter device code.
