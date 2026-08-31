# Gram Newton-Schulz for Ampere

CUDA-only Newton-Schulz orthogonalization with production Muon and Dion3
optimizers for PyTorch. The package supports local CUDA tensors, NCCL DDP
process groups, and one active CUDA DTensor shard dimension.

## Installation

Install the default Torch/cuBLAS backend:

```bash
uv add "gram-newton-schulz-ampere @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@ampere-muon"
```

Install an optional GPU backend:

```bash
uv add "gram-newton-schulz-ampere[cutlass] @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@ampere-muon"
uv add "gram-newton-schulz-ampere[triton] @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@ampere-muon"
```

The CUTLASS backend uses `flashinfer-python==0.6.13` to JIT-compile its SM80
extension. Report JIT compilation separately from steady-state timing.

## Orthogonalization

```python
import torch
from gram_newton_schulz_ampere import StandardNewtonSchulz

orthogonalize = StandardNewtonSchulz(ns_backend="torch")
matrix = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)
result = orthogonalize(matrix)
```

`StandardNewtonSchulz` is the default optimizer algorithm because it was faster
than Gram Newton-Schulz across the measured A100 shape grid. Use
`GramNewtonSchulz` when that algorithm is required.

Available backends are:

- `torch`: default Torch/cuBLAS implementation.
- `cutlass`: FlashInfer-JIT CUTLASS implementation for aligned SM80 tensors.
  Unsupported shapes and devices use the Torch implementation.
- `triton`: explicit Triton implementation.

## Muon

```python
import torch
from gram_newton_schulz_ampere import Muon

weight = torch.nn.Parameter(torch.randn((128, 256), device="cuda"))
optimizer = Muon(
    [weight],
    lr=3e-3,
    ns_algorithm="standard_newton_schulz",
    ns_backend="torch",
)
```

Muon routes matrix parameters through Newton-Schulz. AdamW or Lion parameter
groups handle vectors and scalars.

## Dion3

```python
import torch
from gram_newton_schulz_ampere import Dion3

matrix = torch.nn.Parameter(torch.randn((128, 256), device="cuda"))
vector = torch.nn.Parameter(torch.zeros(128, device="cuda"))
optimizer = Dion3(
    (
        {"params": [matrix], "algorithm": "dion3"},
        {"params": [vector], "algorithm": "adamw"},
    ),
    lr=3e-3,
    fraction=0.25,
    momentum=0.95,
    muon_beta2=0.95,
)
```

Dion3 implements row selection, error-feedback momentum, and FP32 NorMuon
per-neuron normalization. On first use, it builds handwritten `sm_80` and `sm_86`
CUDA kernels plus batched C++ entry points for momentum and decay, row
selection, normalization, and update application. `selection_scope="local"` is
currently supported. Fractional row-sharded selection and norm preservation are
therefore local to each shard and layout-dependent.

## Distributed layouts

Muon supports replicated NCCL DDP parameters and one active CUDA FSDP2/DTensor
matrix shard. Dion3 supports replicated parameters, batched matrices, and one
active row shard. The optimizers reject `Partial` placements, multiple active shard
dimensions, and unsupported Dion3 column shards.

Optimizer state supports `state_dict`, schedulers, `add_param_group`, and PyTorch
distributed-checkpoint conversion. Newton-Schulz settings are included in
optimizer checkpoints.

## Attribution

The Newton-Schulz implementation is based on
[Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz).
The distributed optimizer design and Dion3 algorithm are adapted from
[Microsoft Dion](https://github.com/microsoft/dion).

```bibtex
@misc{GramNewtonSchulz,
  title  = {Gram Newton-Schulz},
  author = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
  year   = {2026},
  url    = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
}
```
