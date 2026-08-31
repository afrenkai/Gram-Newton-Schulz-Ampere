from types import ModuleType
from typing import Protocol

import torch
from torch import Tensor

try:
    from .kernels import triton_ns as triton_kernels
except ImportError:
    triton_kernels = None

from .kernels import cutlass_ns as cutlass_kernels


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


class CutlassBackend:
    def __init__(self, fallback: MatrixBackend) -> None:
        self.fallback = fallback

    def is_candidate(self, tensor: Tensor) -> bool:
        return (
            cutlass_kernels.cutlass_is_installed()
            and tensor.is_cuda
            and torch.cuda.get_device_capability(tensor.device)[0] == 8
        )

    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        if not self.is_candidate(left):
            return self.fallback.symmetric_matmul(left, right)
        try:
            return cutlass_kernels.cutlass_bmm(left, right)
        except ValueError:
            return self.fallback.symmetric_matmul(left, right)

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1,
        beta: float = 1,
    ) -> Tensor:
        if not self.is_candidate(left):
            return self.fallback.symmetric_batch_matrix_matrix_product(
                left,
                right,
                accumulator,
                alpha,
                beta,
            )
        try:
            return cutlass_kernels.cutlass_baddbmm(
                accumulator,
                left,
                right,
                alpha=alpha,
                beta=beta,
            )
        except ValueError:
            return self.fallback.symmetric_batch_matrix_matrix_product(
                left,
                right,
                accumulator,
                alpha,
                beta,
            )

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        if not self.is_candidate(left):
            return self.fallback.matmul(left, right)
        try:
            return cutlass_kernels.cutlass_bmm(left, right)
        except ValueError:
            return self.fallback.matmul(left, right)

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        if not self.is_candidate(left):
            return self.fallback.matmul_add(left, right, accumulator, beta)
        try:
            return cutlass_kernels.cutlass_baddbmm(
                accumulator,
                left,
                right,
                beta=beta,
            )
        except ValueError:
            return self.fallback.matmul_add(left, right, accumulator, beta)


TORCH_BE: MatrixBackend = TorchBackend()
TRITON_BE: MatrixBackend | None = (
    TritonBackend(triton_kernels) if triton_kernels is not None else None
)
CUTLASS_BE = CutlassBackend(fallback=TORCH_BE)


def select_matrix_backend(name: str) -> MatrixBackend:
    if name == "auto":
        return TORCH_BE
    if name == "cutlass":
        if not cutlass_kernels.cutlass_is_installed():
            raise RuntimeError(
                "ns_backend='cutlass' requires flashinfer-python with JIT support"
            )
        return CUTLASS_BE
    if name == "triton":
        if TRITON_BE is None:
            raise RuntimeError("ns_backend='triton' requires Triton")
        return TRITON_BE
    raise ValueError(f"Unknown Newton--Schulz kernel backend: {name}")
