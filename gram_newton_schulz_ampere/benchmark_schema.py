# ruff: noqa: F722
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float
from torch import Tensor

type MatrixOperation = Callable[
    [Float[Tensor, "batch rows columns"]],
    Float[Tensor, "batch rows columns"],
]


@dataclass(frozen=True)
class BenchmarkConfig:
    shapes: tuple[tuple[int, int], ...]
    batch_sizes: tuple[int, ...]
    coefficients: tuple[tuple[float, float, float], ...]
    reset_iterations: tuple[int, ...]
    operations: tuple[str, ...]
    warmups: int
    repeats: int
    seed: int
    dtype: str
    device: str
    epsilon: float
    compile_operations: bool
    output: Path
