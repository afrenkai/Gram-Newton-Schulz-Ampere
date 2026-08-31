# Gram Newton-Schulz for Ampere Architecture
Currently Tested on 3090 and 3080Ti

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
uv add "gram-newton-schulz-ampere @ git+https://github.com/afrenkai/Gram-Newton-Schulz-Ampere.git"
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
