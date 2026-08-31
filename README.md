# Gram Newton-Schulz for Ampere Architecture
Validated on RTX 3090, RTX 3080 Ti, A100, and H100

Ampere compatible kernels for Gram-Newton-Schulz based off code from:
- https://github.com/Dao-AILab/gram-newton-schulz

```
@misc{GramNewtonSchulz,
  title   = {Gram Newton-Schulz},
  author  = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
  year    = {2026},
  url     = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
}
```
More docs soon

## Installation

```bash
uv add "gram-newton-schulz-ampere @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@ampere-muon"
```

## Gram Newton-Schulz

The package exposes an API compatible with the Hopper and Blackwell implementation:

```python
import torch
from gram_newton_schulz_ampere import GramNewtonSchulz

orthogonalize = GramNewtonSchulz(
    ns_use_kernels=True,
    gram_newton_schulz_reset_iterations=[2],
)
matrix = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)
result = orthogonalize(matrix)
```

## Muon

```python
import torch
from gram_newton_schulz_ampere import Muon

weight = torch.nn.Parameter(torch.randn((128, 256), device="cuda"))
optimizer = Muon(
    [weight],
    lr=3e-3,
    ns_algorithm="gram_newton_schulz",
    ns_use_kernels=True,
)
```

The Muon API is adapted from `Dao-AILab/gram-newton-schulz`. On Ampere,
orthogonalization is routed through the Triton kernels in this repository.
The public API is kept independent of the kernel backend so Triton can later be
replaced by CUTLASS or raw CUDA.

## Dion3

`Dion3` adds row selection, error-feedback momentum, and NorMuon
per-neuron normalization to the same Ampere orthogonalization and distributed
collective layer:

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
    ns_use_kernels=True,
)
```

Muon supports replicated DDP parameters, one active FSDP2/DTensor matrix shard,
and feasible one-dimensional tensor parallel layouts. Dion3 supports replicated
parameters, batched matrices such as convolution kernels, and one active row
shard. Multi-shard FSDP2 plus tensor parallel layouts and `Partial` DTensors are
rejected. Dion3 currently supports `selection_scope="local"`; fractional
row-sharded selection is therefore layout-dependent, as in Dion's local mode.

Optimizer state is eagerly allocated and supports normal `state_dict`, learning
rate schedulers, `add_param_group`, and PyTorch distributed-checkpoint state-dict
conversion. Newton--Schulz settings are stored in optimizer checkpoints.

The distributed optimizer design and Dion3 algorithm are adapted from
[Microsoft Dion](https://github.com/microsoft/dion). The Ampere Gram
Newton--Schulz implementation is based on
[Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz).
