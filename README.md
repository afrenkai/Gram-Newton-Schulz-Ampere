# Gram Newton-Schulz for Ampere

CUDA-based (Gram) Newton-Schulz orthogonalization. 
Also has Muon and Dion3 optimizers for use in PyTorch

Currently supported are local CUDA tenrors, DDP NCCL process groups, and 1 DTensor shard dimension. 



## Installation

The Torch/cuBLAS backend can be installed using git (until I add it to pypi)

```bash
uv add "gram-newton-schulz-ampere @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@main"
```

If you want a (faster) GPU backend, you can install them as follows:

```bash
uv add "gram-newton-schulz-ampere[cutlass] @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@main"
uv add "gram-newton-schulz-ampere[triton] @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git@main"
```

Note on the CUTLASS backend. For JIT compilation, we use `flashinfer-python`, for which the current version is `0.6.13`

## Example Usage (Orthogonalization, no Optimizer)

```python
import torch
from gram_newton_schulz_ampere import GramNewtonSchulz

orthogonalize = GramNewtonSchulz(ns_backend="torch")
matrix = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)
result = orthogonalize(matrix)
```

We use `GramNewtonSchulz` as our default optimizer algoruthm. For the naive torch/cuBLAS implementation, following `gram-newton-schulz` we use static per-shape `torch.compile` if a tensor has dim < 256 and a dynamic compile elsewhere. In either case, we use CUDA graphs to avoid overhead with eager compilation. 
If you would like to see the full outer graph or debug eager mode, pass `ns_compile=False`:

```python
import torch
from gram_newton_schulz_ampere import GramNewtonSchulz

orthogonalize = GramNewtonSchulz(ns_backend="cutlass", ns_compile=False)
matrix = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)
result = orthogonalize(matrix)

```
The available backends are:

- `torch`: default Torch/cuBLAS implementation.
- `cutlass`: FlashInfer-JIT CUTLASS implementation for SM8X tensors.
  Unsupported shapes and devices use the Torch implementation.
- `triton`: Triton implementation.


## Example Usage: Muon

```python
import torch
from gram_newton_schulz_ampere import Muon

weight = torch.nn.Parameter(torch.randn((128, 256), device="cuda"))
optimizer = Muon(
    [weight],
    lr=3e-3,
    ns_algorithm="gram_newton_schulz",
    ns_backend="torch",
)
```

Make sure you throw vectors and scalars into AdamW or Lion or a similar Optimizer to avoid mishandling them. Same backends apply

## Example Usage: Dion3

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
Following its paper, our Ampere version of Dion3 implements row selection, error-feedback momentum, and FP32 NorMuon per-neuron normalization. 

The first time it is compiled, it builds handwritten `sm_80` and `sm_86`
CUDA kernels plus batched C++ entry points for momentum and decay, row
selection, normalization, and update application.

`selection_scope="local"` is currently supported. Fractional row-sharded selection and norm preservation are
therefore local to each shard and layout-dependent.

## Distributed layouts

Muon supports replicated NCCL DDP parameters and one active CUDA FSDP2/DTensor
matrix shard. Dion3 supports replicated parameters, batched matrices, and one
active row shard. The optimizers reject `Partial` placements, multiple active shard
dimensions, and unsupported Dion3 column shards.

Optimizer state supports standard Optimizer functionality: i.e.
- `state_dict`,
- schedulers,
- `add_param_group`

We also support PyTorch distributed-checkpoint conversion for resuming across DDP


## Contributing

File an Issue! Bugs are sure to exist and any and all help is more than welcome. This is a solo project, so I might take a while to get to it. 

## Sources

The Newton-Schulz implementation is based on
[Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz).
The distributed optimizer design and Dion3 algorithm are adapted from
[Microsoft Dion](https://github.com/microsoft/dion).

