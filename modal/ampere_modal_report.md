# Modal Ampere actual-shape benchmark

App: `gns-ampere-symmetric-cutlass` (`ap-u2uScuhP4ydEUAJtMvNcy3`)

All reported timings use the actual `B x 16384 x 2048` BF16 input shape from the
email. They are warmed steady-state medians from 5 warmups and 20 CUDA-event
samples. The input seed is 67, with `YOU_COEFFICIENTS`, `eps=1e-7`, and reset
iteration `[2]`.

“Naive” is the pinned upstream eager `StandardNewtonSchulz` implementation at
`e45d0aca7083cb275c9a303220c05c4abecd9187`, with kernels disabled and no compile
kwargs. “Ours” is the cleaned triangular CUTLASS GNS implementation.

## A100-80GB / SM80

| B | Naive Standard (ms) | Ours (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 7.401 | 3.684 | 2.009x |
| 8 | 55.572 | 25.536 | 2.176x |
| 32 | 243.070 | 100.802 | 2.411x |

## A10 / SM86

| B | Naive Standard (ms) | Ours (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 24.714 | 10.330 | 2.393x |
| 8 | 219.534 | 84.705 | 2.592x |
| 32 | 1036.784 | 380.770 | 2.723x |

The A10 container reported 22.06 GiB of usable GPU memory. B32 completed after the
benchmark computed accuracy batch-wise and released completed outputs between
variants. These changes are outside every timed region.

## Validation

- Reported devices and compute capabilities are `NVIDIA A100-SXM4-80GB` / `[8, 0]`
  and `NVIDIA A10` / `[8, 6]`.
- Every completed actual-shape output is finite.
- Relative error versus Torch GNS is `7.32e-4` to `1.33e-3`.
- Output RMS is `0.007832`.
- Orthogonality residual is `0.035409` to `0.035439`.
- Every promised symmetric output has maximum asymmetry exactly `0.0`.

The A10 and A100 rows are separate within-device comparisons. Raw latency must not
be compared across the two GPU models. The A10 is an SM86 compatibility and
performance test, not an RTX 3090 timing proxy.

The A100 function is now deployed with 8 GiB host RAM to reduce future scheduling
constraints. No H100 job was launched. The obsolete SM86-only app
`ap-fHbyJXTEpJSbPepaO8RrkE` is stopped.

Raw results:

- `modal/a100-sm80-results.json`
- `modal/a10-sm86-results.json`
- `modal/ampere-modal-summary.csv`
