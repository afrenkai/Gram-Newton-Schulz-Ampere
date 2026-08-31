import torch
from torch import Tensor


class TorchBackend:
    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
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
