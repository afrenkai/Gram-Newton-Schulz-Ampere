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


class Dion3CudaExtension(Protocol):
    def momentum_row_norm(self, momentum: Tensor, gradient: Tensor) -> Tensor: ...

    def gather_decay(
        self,
        momentum: Tensor,
        indices: Tensor,
        decay: float,
    ) -> Tensor: ...

    def select_rows(
        self,
        momentum: Tensor,
        gradient: Tensor,
        parameter: Tensor,
        selected_count: int,
        decay: float,
        parameter_decay: float,
    ) -> list[Tensor]: ...

    def select_rows_batch(
        self,
        momenta: list[Tensor],
        gradients: list[Tensor],
        parameters: list[Tensor],
        selected_counts: list[int],
        decay: float,
        parameter_decay: float,
    ) -> list[Tensor]: ...

    def normalize_rows(
        self,
        selected_update: Tensor,
        variance: Tensor,
        indices: Tensor,
        beta_two: float,
        epsilon: float,
    ) -> list[Tensor]: ...

    def apply_rows(
        self,
        parameter: Tensor,
        normalized_update: Tensor,
        indices: Tensor,
        squared_norms: Tensor,
        learning_rate: float,
        weight_decay: float,
        adjusted_learning_rate: float,
        epsilon: float,
    ) -> None: ...

    def normalize_apply_rows(
        self,
        parameter: Tensor,
        selected_update: Tensor,
        variance: Tensor,
        indices: Tensor,
        beta_two: float,
        epsilon: float,
        learning_rate: float,
        weight_decay: float,
        adjusted_learning_rate: float,
    ) -> None: ...

    def normalize_apply_rows_batch(
        self,
        parameters: list[Tensor],
        selected_updates: list[Tensor],
        variances: list[Tensor],
        indices: list[Tensor],
        beta_two: float,
        epsilon: float,
        learning_rate: float,
        weight_decay: float,
        adjusted_learning_rates: list[float],
    ) -> None: ...
