from typing import Protocol

from torch import Tensor


class MatrixBackend(Protocol):
    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor: ...

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor: ...

    def matmul(self, left: Tensor, right: Tensor) -> Tensor: ...

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor: ...
