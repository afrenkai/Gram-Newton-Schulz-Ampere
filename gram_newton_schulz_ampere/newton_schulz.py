# ruff: noqa: F722
from collections.abc import Sequence

import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor

from gram_newton_schulz_ampere.coefficients import polar_express_coefficients
from gram_newton_schulz_ampere.kernels.interface import select_matrix_backend
from gram_newton_schulz_ampere.optimizer_types import NewtonSchulzBackend


class NewtonSchulz:
    def __init__(
        self,
        eps: float = 1e-9,
        coeff: Sequence[Sequence[float]] | None = None,
        use_gram: bool = False,
        gns_reset_iters: Sequence[int] | None = None,
        backend: NewtonSchulzBackend = "torch",
    ) -> None:
        self.epsilon = eps
        self.coefficients = coeff if coeff is not None else polar_express_coefficients()
        self.backend = select_matrix_backend(backend)
        self.gram = use_gram
        self.reset_iterations = tuple(gns_reset_iters or ())

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        matrix: Float[Tensor, "*batch rows columns"],
    ) -> Float[Tensor, "*batch rows columns"]:
        if not matrix.is_cuda:
            raise ValueError("Newton--Schulz requires a CUDA tensor")
        if matrix.ndim < 2:
            raise ValueError("Newton--Schulz expects a matrix or a batch of matrices")
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

        matrix.div_(matrix.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        matrix = matrix.to(torch.float16)
        if not self.gram or matrix.shape[-2] == matrix.shape[-1]:
            matrix = self.standard_iteration(matrix)
        else:
            matrix = self.gram_iteration(matrix)

        if should_transpose:
            matrix = matrix.mT
        return matrix.to(original_dtype).reshape(original_shape)

    def standard_iteration(self, matrix: Tensor) -> Tensor:
        for coefficient_one, coefficient_two, coefficient_three in self.coefficients:
            gram_matrix = self.backend.symmetric_matmul(matrix, matrix.mT)
            polynomial = self.backend.symmetric_batch_matrix_matrix_product(
                gram_matrix,
                gram_matrix,
                accumulator=gram_matrix,
                alpha=coefficient_three,
                beta=coefficient_two,
            )
            matrix = self.backend.matmul_add(
                polynomial,
                matrix,
                accumulator=matrix,
                beta=coefficient_one,
            )
        return matrix

    def gram_iteration(self, matrix: Tensor) -> Tensor:
        gram_matrix = self.backend.symmetric_matmul(matrix, matrix.mT)
        identity = torch.eye(
            gram_matrix.shape[-1],
            device=matrix.device,
            dtype=matrix.dtype,
        ).unsqueeze(0)
        accumulated_polynomial = None
        for iteration, coefficients in enumerate(self.coefficients):
            coefficient_one, coefficient_two, coefficient_three = coefficients
            if iteration != 0 and iteration in self.reset_iterations:
                if accumulated_polynomial is None:
                    raise RuntimeError(
                        "Gram Newton--Schulz reset has no accumulated update"
                    )
                matrix = self.backend.matmul(accumulated_polynomial, matrix)
                gram_matrix = self.backend.symmetric_matmul(matrix, matrix.mT)
                accumulated_polynomial = None

            polynomial = self.backend.symmetric_batch_matrix_matrix_product(
                gram_matrix,
                gram_matrix,
                accumulator=gram_matrix,
                alpha=coefficient_three,
                beta=coefficient_two,
            )
            if accumulated_polynomial is None:
                accumulated_polynomial = polynomial.add(
                    identity,
                    alpha=coefficient_one,
                )
            else:
                accumulated_polynomial = (
                    self.backend.symmetric_batch_matrix_matrix_product(
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
                gram_polynomial = self.backend.symmetric_batch_matrix_matrix_product(
                    gram_matrix,
                    polynomial,
                    accumulator=gram_matrix,
                    beta=coefficient_one,
                )
                gram_matrix = self.backend.symmetric_batch_matrix_matrix_product(
                    polynomial,
                    gram_polynomial,
                    accumulator=gram_polynomial,
                    beta=coefficient_one,
                )
        if accumulated_polynomial is None:
            raise RuntimeError("Gram Newton--Schulz requires coefficients")
        return self.backend.matmul(accumulated_polynomial, matrix)


class GramNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_backend: NewtonSchulzBackend = "torch",
        ns_coefficients: Sequence[Sequence[float]] | None = None,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=True,
            gns_reset_iters=gram_newton_schulz_reset_iterations,
            backend=ns_backend,
        )


class StandardNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_backend: NewtonSchulzBackend = "torch",
        ns_coefficients: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=False,
            backend=ns_backend,
        )
