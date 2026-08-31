# ruff: noqa: F722
from collections.abc import Callable, Sequence

import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor

from gram_newton_schulz_ampere.coefficients import polar_express_coefficients
from gram_newton_schulz_ampere.kernels.interface import select_matrix_backend
from gram_newton_schulz_ampere.kernels.types import MatrixBackend
from gram_newton_schulz_ampere.optimizer_types import NewtonSchulzBackend


def compile_newton_schulz_operation(
    operation: Callable[[Tensor], Tensor],
    enabled: bool,
    dynamic: bool,
) -> Callable[[Tensor], Tensor]:
    if not enabled:
        return operation
    if dynamic:
        compiled_operation = torch.compile(
            operation,
            fullgraph=True,
            dynamic=True,
            options={
                "max_autotune_gemm_backends": "ATEN",
                "triton.cudagraphs": True,
            },
        )
    else:
        compiled_operation = torch.compile(
            operation,
            fullgraph=True,
            mode="reduce-overhead",
        )

    @torch.compiler.config.patch(  # ty: ignore[unresolved-attribute]
        recompile_limit=256
    )
    def operation_with_recompile_limit(matrix: Tensor) -> Tensor:
        return compiled_operation(matrix)

    return operation_with_recompile_limit


def build_standard_newton_schulz_operation(
    backend: MatrixBackend,
    coefficients: Sequence[Sequence[float]],
    epsilon: float,
    compile_operation: bool,
    dynamic_compile: bool,
    normalize_input: bool,
) -> Callable[[Tensor], Tensor]:
    coefficient_values = tuple(
        tuple(float(coefficient) for coefficient in iteration_coefficients)
        for iteration_coefficients in coefficients
    )

    def standard_newton_schulz(matrix: Tensor) -> Tensor:
        tall_skinny = matrix.size(-2) > matrix.size(-1)
        if normalize_input:
            matrix = matrix.to(torch.float32)
            matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + epsilon)
            matrix = matrix.to(torch.float16)
        for coefficient_one, coefficient_two, coefficient_three in coefficient_values:
            if tall_skinny:
                gram_matrix = backend.symmetric_matmul(matrix.mT, matrix)
            else:
                gram_matrix = backend.symmetric_matmul(matrix, matrix.mT)
            polynomial = backend.symmetric_batch_matrix_matrix_product(
                gram_matrix,
                gram_matrix,
                accumulator=gram_matrix,
                alpha=coefficient_three,
                beta=coefficient_two,
            )
            if tall_skinny:
                matrix = backend.matmul_add(
                    matrix,
                    polynomial,
                    accumulator=matrix,
                    beta=coefficient_one,
                )
            else:
                matrix = backend.matmul_add(
                    polynomial,
                    matrix,
                    accumulator=matrix,
                    beta=coefficient_one,
                )
        return matrix

    return compile_newton_schulz_operation(
        standard_newton_schulz,
        enabled=compile_operation,
        dynamic=dynamic_compile,
    )


def build_gram_newton_schulz_operation(
    backend: MatrixBackend,
    coefficients: Sequence[Sequence[float]],
    reset_iterations: Sequence[int],
    epsilon: float,
    compile_operation: bool,
    dynamic_compile: bool,
    normalize_input: bool,
) -> Callable[[Tensor], Tensor]:
    coefficient_values = tuple(
        tuple(float(coefficient) for coefficient in iteration_coefficients)
        for iteration_coefficients in coefficients
    )
    reset_iteration_values = frozenset(reset_iterations)

    def gram_newton_schulz(matrix: Tensor) -> Tensor:
        tall_skinny = matrix.size(-2) > matrix.size(-1)
        if normalize_input:
            matrix = matrix.to(torch.float32)
            matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + epsilon)
            matrix = matrix.to(torch.float16)
        if tall_skinny:
            gram_matrix = backend.symmetric_matmul(matrix.mT, matrix)
        else:
            gram_matrix = backend.symmetric_matmul(matrix, matrix.mT)
        identity = (
            torch.eye(
                gram_matrix.shape[-1],
                device=matrix.device,
                dtype=matrix.dtype,
            )
            .unsqueeze(0)
            .expand(gram_matrix.shape[0], -1, -1)
            .contiguous()
        )
        accumulated_polynomial = None
        for iteration, coefficients_for_iteration in enumerate(coefficient_values):
            coefficient_one, coefficient_two, coefficient_three = (
                coefficients_for_iteration
            )
            if iteration != 0 and iteration in reset_iteration_values:
                if accumulated_polynomial is None:
                    raise RuntimeError(
                        "Gram Newton--Schulz reset has no accumulated update"
                    )
                if tall_skinny:
                    matrix = backend.matmul(matrix, accumulated_polynomial)
                    gram_matrix = backend.symmetric_matmul(matrix.mT, matrix)
                else:
                    matrix = backend.matmul(accumulated_polynomial, matrix)
                    gram_matrix = backend.symmetric_matmul(matrix, matrix.mT)
                accumulated_polynomial = None

            polynomial = backend.symmetric_batch_matrix_matrix_product(
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
                accumulated_polynomial = backend.symmetric_batch_matrix_matrix_product(
                    accumulated_polynomial,
                    polynomial,
                    accumulator=accumulated_polynomial,
                    beta=coefficient_one,
                )
            if (
                iteration < len(coefficient_values) - 1
                and iteration + 1 not in reset_iteration_values
            ):
                gram_polynomial = backend.symmetric_batch_matrix_matrix_product(
                    gram_matrix,
                    polynomial,
                    accumulator=gram_matrix,
                    beta=coefficient_one,
                )
                gram_matrix = backend.symmetric_batch_matrix_matrix_product(
                    polynomial,
                    gram_polynomial,
                    accumulator=gram_polynomial,
                    beta=coefficient_one,
                )
        if accumulated_polynomial is None:
            raise RuntimeError("Gram Newton--Schulz requires coefficients")
        if tall_skinny:
            return backend.matmul(matrix, accumulated_polynomial)
        return backend.matmul(accumulated_polynomial, matrix)

    return compile_newton_schulz_operation(
        gram_newton_schulz,
        enabled=compile_operation,
        dynamic=dynamic_compile,
    )


class NewtonSchulz:
    def __init__(
        self,
        eps: float = 1e-9,
        coeff: Sequence[Sequence[float]] | None = None,
        use_gram: bool = False,
        gns_reset_iters: Sequence[int] | None = None,
        backend: NewtonSchulzBackend = "torch",
        compile: bool = True,
    ) -> None:
        self.epsilon = eps
        self.coefficients = coeff if coeff is not None else polar_express_coefficients()
        self.backend = select_matrix_backend(backend)
        self.gram = use_gram
        self.reset_iterations = tuple(gns_reset_iters or ())
        compile_operation = compile and backend == "torch"
        self.compile_operation = compile_operation
        self.standard_operation = build_standard_newton_schulz_operation(
            backend=self.backend,
            coefficients=self.coefficients,
            epsilon=self.epsilon,
            compile_operation=compile_operation,
            dynamic_compile=False,
            normalize_input=True,
        )
        self.gram_operation = build_gram_newton_schulz_operation(
            backend=self.backend,
            coefficients=self.coefficients,
            reset_iterations=self.reset_iterations,
            epsilon=self.epsilon,
            compile_operation=compile_operation,
            dynamic_compile=False,
            normalize_input=True,
        )
        if compile_operation:
            self.dynamic_standard_operation = build_standard_newton_schulz_operation(
                backend=self.backend,
                coefficients=self.coefficients,
                epsilon=self.epsilon,
                compile_operation=True,
                dynamic_compile=True,
                normalize_input=False,
            )
            self.dynamic_gram_operation = build_gram_newton_schulz_operation(
                backend=self.backend,
                coefficients=self.coefficients,
                reset_iterations=self.reset_iterations,
                epsilon=self.epsilon,
                compile_operation=True,
                dynamic_compile=True,
                normalize_input=False,
            )
        else:
            self.dynamic_standard_operation = self.standard_operation
            self.dynamic_gram_operation = self.gram_operation

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
        dynamic_compile = (
            self.compile_operation and min(matrix.shape[-2], matrix.shape[-1]) >= 256
        )
        if dynamic_compile:
            matrix = matrix.to(torch.float32)
            matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
            matrix = matrix.to(torch.float16)
        if self.gram and matrix.shape[-2] != matrix.shape[-1]:
            operation = (
                self.dynamic_gram_operation if dynamic_compile else self.gram_operation
            )
        else:
            operation = (
                self.dynamic_standard_operation
                if dynamic_compile
                else self.standard_operation
            )
        matrix = operation(matrix)
        return matrix.to(original_dtype).reshape(original_shape)


class GramNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_backend: NewtonSchulzBackend = "torch",
        ns_coefficients: Sequence[Sequence[float]] | None = None,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
        ns_compile: bool = True,
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=True,
            gns_reset_iters=gram_newton_schulz_reset_iterations,
            backend=ns_backend,
            compile=ns_compile,
        )


class StandardNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_backend: NewtonSchulzBackend = "torch",
        ns_coefficients: Sequence[Sequence[float]] | None = None,
        ns_compile: bool = True,
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=ns_coefficients,
            use_gram=False,
            backend=ns_backend,
            compile=ns_compile,
        )
