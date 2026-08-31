# ruff: noqa: F722
from collections.abc import Sequence

import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .ampere_ns_interface import TORCH_BE, MatrixBackend, select_matrix_backend
from .coefficients import POLAR_EXPRESS_COEFFICIENTS
from .muon_types import NewtonSchulzBackend


class NewtonSchulz:
    def __init__(
        self,
        eps: float = 1e-9,
        coeff: Sequence[Sequence[float]] | None = None,
        use_gram: bool = False,
        use_triton: bool = False,
        gns_reset_iters: Sequence[int] | None = None,
        kernel_backend: NewtonSchulzBackend = "auto",
    ) -> None:
        self.epsilon = eps
        self.coefficients = coeff if coeff is not None else POLAR_EXPRESS_COEFFICIENTS
        self.backend: MatrixBackend = (
            select_matrix_backend(kernel_backend) if use_triton else TORCH_BE
        )
        self.gram = use_gram
        self.reset_iterations = tuple(gns_reset_iters or ())

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        matrix: Float[Tensor, "*batch rows columns"],
    ) -> Float[Tensor, "*batch rows columns"]:
        original_shape = matrix.shape
        if matrix.ndim == 2:
            matrix = rearrange(matrix, "rows columns -> 1 rows columns")
        elif matrix.ndim > 3:
            matrix = rearrange(
                matrix,
                "... rows columns -> (...) rows columns",
            )

        original_dtype = matrix.dtype
        matrix = matrix.to(torch.float32)
        should_transpose = matrix.shape[-2] > matrix.shape[-1]
        if should_transpose:
            matrix = matrix.mT

        matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        matrix = matrix.to(torch.float16)
        backend = self.backend
        if not self.gram or matrix.shape[-2] == matrix.shape[-1]:
            matrix = self.standard_iteration(matrix, backend)
        else:
            matrix = self.gram_iteration(matrix, backend)

        if should_transpose:
            matrix = matrix.mT
        return matrix.to(original_dtype).reshape(original_shape)

    def standard_iteration(
        self,
        matrix: Tensor,
        backend: MatrixBackend | None = None,
    ) -> Tensor:
        selected_backend = self.backend if backend is None else backend
        for coefficient_one, coefficient_two, coefficient_three in self.coefficients:
            gram_matrix = selected_backend.symmetric_matmul(matrix, matrix.mT)
            polynomial = selected_backend.symmetric_batch_matrix_matrix_product(
                gram_matrix,
                gram_matrix,
                accumulator=gram_matrix,
                alpha=coefficient_three,
                beta=coefficient_two,
            )
            matrix = selected_backend.matmul_add(
                polynomial,
                matrix,
                accumulator=matrix,
                beta=coefficient_one,
            )
        return matrix

    def gram_iteration(
        self,
        matrix: Tensor,
        backend: MatrixBackend | None = None,
    ) -> Tensor:
        selected_backend = self.backend if backend is None else backend
        gram_matrix = selected_backend.symmetric_matmul(matrix, matrix.mT)
        batch_size = gram_matrix.shape[0]
        identity = (
            torch.eye(
                gram_matrix.shape[-1],
                device=matrix.device,
                dtype=matrix.dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        accumulated_polynomial = None
        for iteration, coefficients in enumerate(self.coefficients):
            coefficient_one, coefficient_two, coefficient_three = coefficients
            if iteration != 0 and iteration in self.reset_iterations:
                if accumulated_polynomial is None:
                    raise RuntimeError(
                        "Gram Newton--Schulz reset has no accumulated update"
                    )
                matrix = selected_backend.matmul(accumulated_polynomial, matrix)
                gram_matrix = selected_backend.symmetric_matmul(matrix, matrix.mT)
                accumulated_polynomial = None

            polynomial = selected_backend.symmetric_batch_matrix_matrix_product(
                gram_matrix,
                gram_matrix,
                accumulator=gram_matrix,
                alpha=coefficient_three,
                beta=coefficient_two,
            )
            if accumulated_polynomial is None:
                accumulated_polynomial = polynomial + coefficient_one * identity
            else:
                accumulated_polynomial = (
                    selected_backend.symmetric_batch_matrix_matrix_product(
                        accumulated_polynomial,
                        polynomial,
                        accumulator=accumulated_polynomial,
                        beta=coefficient_one,
                    )
                )
            if (
                iteration < len(self.coefficients) - 1
                and iteration + 1 not in self.reset_iterations
            ):
                gram_polynomial = (
                    selected_backend.symmetric_batch_matrix_matrix_product(
                        gram_matrix,
                        polynomial,
                        accumulator=gram_matrix,
                        beta=coefficient_one,
                    )
                )
                gram_matrix = selected_backend.symmetric_batch_matrix_matrix_product(
                    polynomial,
                    gram_polynomial,
                    accumulator=gram_polynomial,
                    beta=coefficient_one,
                )
        if accumulated_polynomial is None:
            raise RuntimeError("Gram Newton--Schulz requires coefficients")
        return selected_backend.matmul(accumulated_polynomial, matrix)


class GramNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_backend: NewtonSchulzBackend = "auto",
        ns_coefficients: Sequence[Sequence[float]] = POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=True,
            use_triton=ns_use_kernels,
            gns_reset_iters=gram_newton_schulz_reset_iterations,
            kernel_backend=ns_backend,
        )


class StandardNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_backend: NewtonSchulzBackend = "auto",
        ns_coefficients: Sequence[Sequence[float]] = POLAR_EXPRESS_COEFFICIENTS,
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=False,
            use_triton=ns_use_kernels,
            kernel_backend=ns_backend,
        )
