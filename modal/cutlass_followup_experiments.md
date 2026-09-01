# Ampere CUTLASS follow-up experiments

## Outcome

The separate 32-by-32 mirror kernel remains the production path. The same-CTA
fused-mirror prototype was correct on SM80 and SM86, but it was slower in nearly
all measured configurations. Its implementation is preserved in
`modal/fused_mirror_prototype.patch` (SHA-256 `5b990e9d577bd3864f3e24a8fc49fe637764af61d7798b185f354e07a013b00a`) rather than in the
production JIT source.

All measurements used one GPU. `batch_size` is the number of matrices in one
operator call, not a GPU count.

## Provenance and protocol

- Source baseline: commit `4caa01d8d1875277dbbf7df04fe188c783e0df81` on
  `refactor/ampere-symmetric-cutlass-cleanup`.
- Experimental fused source: `modal/fused_mirror_prototype.patch`.
- A100: NVIDIA A100-SXM4-80GB, SM80, Turing Slurm jobs `2241751`--`2241755`.
- A10: NVIDIA A10, SM86, Modal app
  `ap-Ajaf693bTkmnVZvPoeyrTt`.
- Software for performance runs: PyTorch 2.11.0+cu130, CUDA runtime
  13.0.
- Timings are warmed CUDA-event medians unless explicitly identified as process
  wall time or sampled power data.

Raw JSON, scheduler logs, raw power samples, and derived CSVs are listed in the
artifact section. Results from separate jobs are not pooled as repeated samples.

## Device tests

The experimental revision passed 28 cases on each tested Ampere GPU. After the
prototype was removed, the remaining production suite passed 25/25 on A100.

| Revision | Device | Compute capability | Run | Result |
|---|---|---|---|---:|
| Fused experiment | NVIDIA A100-SXM4-80GB | SM80 | Slurm `2241751` | 28/28 |
| Fused experiment | NVIDIA A10 | SM86 | Modal `ap-Ajaf693bTkmnVZvPoeyrTt` | 28/28 |
| Restored production | NVIDIA A100-SXM4-80GB | SM80 | Slurm `2241769` | 25/25 |

Coverage includes FP16 and BF16, RR/CR/RC layouts, partial tiles, all three full
GEMM tactics, alignment and batch-stride rejection, CPU and irregular-layout
fallbacks, exact symmetry, and a deliberately false symmetry contract. The
false-contract test confirms that the direct symmetric API reflects its computed
upper triangle and therefore produces the wrong full product when the caller's
symmetry promise is false.

## Dimension and aspect-ratio sweep

Jobs `2241752` and `2241770` measured batch one at 19 shapes from
256-by-256 through 16384-by-2048, including square, tall, and wide orientations.
Each method used 3 warmups and 10 measured repeats. All outputs were finite. The
maximum recorded relative error was 0.002342.

| Input rows x columns | Torch Gram (ms) | CUTLASS Gram (ms) | Gram speedup | Standard NS (ms) | Torch GNS (ms) | CUTLASS GNS (ms) | Standard/CUTLASS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 x 256 | 0.050 | 0.053 | 0.942x | 0.586 | 0.634 | 0.613 | 0.956x |
| 4096 x 1024 | 0.083 | 0.098 | 0.844x | 0.819 | 0.806 | 0.840 | 0.976x |
| 4096 x 2048 | 0.196 | 0.162 | 1.205x | 2.375 | 2.200 | 1.921 | 1.236x |
| 8192 x 2048 | 0.314 | 0.267 | 1.176x | 3.678 | 2.907 | 2.591 | 1.419x |
| 16384 x 2048 | 0.552 | 0.450 | 1.229x | 7.048 | 4.069 | 3.642 | 1.935x |
| 2048 x 16384 | 0.560 | 0.466 | 1.202x | 7.102 | 4.104 | 3.697 | 1.921x |

The triangular kernel is not beneficial for small symmetric dimensions. Launch,
mirror, and fixed-tile costs dominate there. Square Gram speedup was 0.718x at
512, 0.763x at 1024, and 1.005x at 2048. Square end-to-end speedup over Standard
was 0.751x, 0.958x, and 1.011x, respectively. The crossover is therefore near a
2048 symmetric dimension in this sweep. Tall and wide forms converge to similar
gains at the original 2048 symmetric dimension. This is empirical policy
evidence, not a general crossover theorem.

## Fused mirror experiment

Job `2241753` produced 216 records over dimensions 256, 512, 1024, and 2048;
inner/output ratios 1, 2, 4, and 8; RR, CR, and RC layouts; and batches 1 and 8.
Every fused result was bitwise equal to the separate-mirror result and exactly
symmetric. Maximum error relative to the full Torch product was
0.000371.

The prototype performs the mirror in the same CTA after the CUTLASS epilogue.
It avoids the second kernel launch, but its transpose reads are strided and one
GEMM CTA performs all mirror work for its 128-by-128 output tile. The measured
fused/separate median ratio was 1.17--1.33 across batch/layout groups. The best
single record was 0.982, while the worst was 1.439. One isolated 1.8 percent win
is not enough to offset the broad regression. This evidence rejects the
prototype as a production optimization.

## Cold JIT cache

Jobs `2241754` and `2241771` each used two independent empty FlashInfer
workspaces, followed by two new processes reusing the first workspace. They
measured a 2048-by-16384 symmetric product. Job `2241754` included the rejected
fused template; job `2241771` used the restored production source.

| Revision and cache state | Process wall time (ms) | First CUTLASS call (ms) | Warm operation median (ms) | Cache size (MB) |
|---|---:|---:|---:|---:|
| Prototype, cold mean | 33094.47 | 30104.71 | 0.504 | 2.56 |
| Prototype, reused mean | 3007.33 | 25.54 | 0.505 | 2.56 |
| Production, cold mean | 26883.49 | 23977.94 | 0.504 | 1.87 |
| Production, reused mean | 3079.91 | 24.08 | 0.502 | 1.87 |

Removing the rejected template reduced mean cold first-call time by 20.4 percent,
whole-process cold time by 18.8 percent, and cached bytes by 27.1 percent. Warm
operation latency did not regress. The first-call value measures compilation and
loading inside the worker; process wall time also includes interpreter and
framework startup. A reused cache removes compilation but does not remove Python
startup or dynamic-library loading.

## Power and utilization

Job `2241755` sampled `nvidia-smi` every 50 ms during one five-second interval
per operation at batch one and input 16384-by-2048. Each operation was warmed
first and synchronized once per measured iteration.

| Operation | Mean latency (ms) | Mean power (W) | Mean GPU util. | Estimated J/iteration |
|---|---:|---:|---:|---:|
| Torch symmetric Gram | 0.566 | 483.74 | 89.55% | 0.274 |
| CUTLASS separate mirror | 0.444 | 406.36 | 88.72% | 0.181 |
| CUTLASS fused mirror | 0.461 | 401.81 | 89.92% | 0.185 |
| Torch GNS | 4.098 | 469.72 | 94.14% | 1.925 |
| CUTLASS GNS | 3.663 | 421.44 | 95.60% | 1.544 |
| Standard NS | 7.130 | 487.17 | 96.71% | 3.474 |

Relative to Torch Gram, the separate-mirror primitive completed 1.274 times as
many iterations, used 16.0 percent less sampled mean power, and used 34.1 percent
less estimated energy per iteration. Relative to Torch GNS, CUTLASS GNS completed
1.118 times as many iterations and used 19.8 percent less estimated energy per
iteration. Relative to Standard NS, it completed 1.944 times as many iterations
and used 55.6 percent less estimated energy per iteration.

These are coarse board-level samples from one fixed-order run. The energy values
are mean sampled power multiplied by elapsed time and divided by completed
iterations. They are useful diagnostic measurements, not calibrated energy or
causal profiler evidence.

## Final production speed regression

After removing the fused prototype, Turing job `2241930` reran the original
batch 1, 8, and 32 end-to-end protocol with 5 warmups and 20 measured repeats.
All outputs were finite, maximum recorded relative error was 0.001887, and all
reported symmetric primitives had zero maximum asymmetry.

| Batch | Standard NS (ms) | Torch GNS (ms) | CUTLASS GNS (ms) | Standard/CUTLASS | CUTLASS change from job 2241582 |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.071 | 4.058 | 3.622 | 1.952x | -0.34% |
| 8 | 52.331 | 30.341 | 24.229 | 2.160x | -0.76% |
| 32 | 223.898 | 129.620 | 95.559 | 2.343x | -0.50% |

All three methods were slightly faster than in job `2241582`, so the small
cross-job changes are consistent with run-level variation rather than a
CUTLASS-specific regression. Within job `2241930`, CUTLASS GNS was 1.120x,
1.252x, and 1.356x faster than Torch GNS at batches 1, 8, and 32.

## Next optimization opportunities

1. Add shape-aware dispatch or a small first-use autotuner. Batch-one cases below
a 2048 symmetric dimension often favor Torch, but batch eight already favors
CUTLASS at some 512 and 1024 cases. A single dimension threshold would therefore
be too coarse; batch, reduction dimension, layout, and dtype should enter the
policy.
2. Prebuild or prewarm the FlashInfer module in deployment images. Even the
restored production source takes about 24 seconds on an empty cache, while a
reused cache loads the first call in about 24 ms.
3. If mirror fusion is revisited, write the transposed output directly from
accumulator fragments or a coalesced shared-memory epilogue. The rejected
prototype rereads row-major global output with strided transpose accesses, so it
does not rule out a materially different dual-output epilogue.
4. Autotune triangular threadblock and warp shapes for symmetric dimensions 512
and 1024. The current triangular path has one fixed 128-by-128 tile while the
full GEMM path already has three tactics.
5. Repeat power runs with randomized operation order, fixed clocks, multiple
processes, and Nsight Compute counters before making hardware-level causal
claims.

## Artifacts

- `modal/dimension-sweep-2241752.json`
- `modal/dimension-square-2241770.json`
- `modal/dimension-sweep-summary-2241752.csv` (combined jobs `2241752` and `2241770`)
- `modal/fused-mirror-2241753.json`
- `modal/fused-mirror-summary-2241753.csv`
- `modal/fused_mirror_prototype.patch`
- `modal/cold-cache-2241754.json` (prototype)
- `modal/cold-cache-2241771.json` (restored production)
- `modal/cold-cache-summary.csv` (both revisions, individual trials)
- `modal/cold-cache-comparison.csv`
- `modal/power-utilization-2241755.json`
- `modal/power-utilization-summary-2241755.csv`
- `modal/power-samples-2241755/`
- `modal/sm8x-validation-summary.csv`
- `modal/symmetric-cutlass-results-2241930.json`
- `modal/final-production-speed-2241930.csv`

The corresponding harnesses are `modal/benchmark_dimension_sweep.py`,
`modal/benchmark_fused_mirror.py`, `modal/benchmark_cold_cache.py`,
`modal/benchmark_power_utilization.py`, `modal/cutlass_test_app.py`, and
`modal/turing_cutlass_experiments.sbatch`. Reproducing the rejected fused run
requires applying `modal/fused_mirror_prototype.patch` first.
