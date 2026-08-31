from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT

type Coefficients = Sequence[Sequence[float]]
type DistributedMesh = DeviceMesh | ProcessGroup | None
type LearningRate = float | Tensor
type LearningRateAdjustment = Literal["spectral_norm", "rms_norm"] | None
type LossClosure = Callable[[], float]
type NewtonSchulzAlgorithm = Literal["gram_newton_schulz", "standard_newton_schulz"]
type NewtonSchulzBackend = Literal["torch", "cutlass", "triton"]
type OptimizerAlgorithm = Literal["muon", "dion3", "adamw", "lion"]
type ParameterGroup = dict[str, object]
type ParameterGroupInput = Mapping[str, object]
type OptimizerParameters = ParamsT


@dataclass(frozen=True)
class ParameterLayout:
    process_group: ProcessGroup | None
    sharded_tensor_dimension: int | None
    batch_sharded: bool


class Orthogonalizer(Protocol):
    def __call__(self, matrix: Tensor, /) -> Tensor: ...
