from types import ModuleType
from typing import Protocol

import torch
from torch import Tensor

try:
    from .kernels import triton_ns as triton_kernels
except ImportError:
    triton_kernels = None


class MatrixBackend(Protocol):
    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor: ...

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1,
        beta: float = 1,
    ) -> Tensor: ...

    def matmul(self, left: Tensor, right: Tensor) -> Tensor: ...

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor: ...


class TorchBackend:
    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1,
        beta: float = 1,
    ) -> Tensor:
        return torch.baddbmm(
            accumulator,
            left,
            right,
            alpha=alpha,
            beta=beta,
        )

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        return torch.baddbmm(accumulator, left, right, beta=beta)


class TritonBackend:
    def __init__(self, kernels: ModuleType) -> None:
        self.kernels = kernels

    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1,
        beta: float = 1,
    ) -> Tensor:
        return self.kernels.triton_baddbmm(
            accumulator,
            left,
            right,
            alpha=alpha,
            beta=beta,
        )

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        return self.kernels.triton_baddbmm(
            accumulator,
            left,
            right,
            beta=beta,
        )


TORCH_BE: MatrixBackend = TorchBackend()
TRITON_BE: MatrixBackend | None = (
    TritonBackend(triton_kernels) if triton_kernels is not None else None
)
