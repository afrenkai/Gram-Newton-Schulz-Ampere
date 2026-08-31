import torch
from torch import Tensor
import pytest
from typing import Generator

class IdentityOrthogonalizer:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def __call__(self, matrix: Tensor) -> Tensor:
        self.input_shapes.append(tuple(matrix.shape))
        return matrix


@pytest.fixture(autouse=True)
def cuda_default_device() -> Generator[None, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("cuda"):
        yield


