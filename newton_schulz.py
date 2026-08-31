from typing import Sequence
import torch
from coefficients import POLAR_EXPRESS_COEFFICIENTS
from ampere_ns_interface import TORCH_BE, TRITON_BE


class NewtonSchulz:
    def __init__(
        self,
        eps: float = 1e-9,
        coeff: list[list[float]] | None = None,
        use_gram: bool = False,
        use_triton: bool = False,
        gns_reset_iters: list[int] | None = None,
        compile_kwargs: dict[str, bool | str] | None = None,
    ) -> None:
        self.eps = eps
        self.coeff = coeff if coeff is not None else POLAR_EXPRESS_COEFFICIENTS
        self.ops = TRITON_BE if use_triton else TORCH_BE
        self.use_gram = use_gram
        self.use_triton = use_triton
        self.gns_reset_iters = gns_reset_iters
        if compile_kwargs is not None and not use_triton:
            self.__call__ = torch.compile(self.__call__, **compile_kwargs)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:

        original_shape = X.size()
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])

        original_dtype = X.dtype
        X = X.to(torch.float32)

        if should_transpose := (X.size(-2) > X.size(-1)):
            X = X.mT

        X /= X.norm(dim=(-2, -1), keepdim=True) + self.eps
        X = X.to(torch.float16)
        if not self.use_gram:
            X = self._newton_schulz(X)
        else:
            X = self._gram_newton_schulz(X)

        if should_transpose:
            X = X.mT
        return X.to(original_dtype).view(original_shape)

    def _newton_schulz(self, X: torch.Tensor) -> torch.Tensor:
        for (
            a,
            b,
            c,
        ) in self.coeff:
            A = self.ops.symmetric_matmul(X, X.mT)
            B = self.ops.symmetric_batch_matrix_matrix_product(
                A, A, C=A, alpha=c, beta=b
            )
            X = self.ops.matmul_add(B, X, C=X, beta=a)
        return X

    def _gram_newton_schulz(self, X: torch.Tensor) -> torch.Tensor:
        R = self.ops.symmetric_matmul(X, X.mT)

        batch_size = R.size(0)
        I = (
            torch.eye(R.size(-1), device=X.device, dtype=X.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        Q = None
        reset_iterations = self.gns_reset_iters or []
        for iteration, (a, b, c) in enumerate(self.coeff):
            if iteration != 0 and iteration in reset_iterations:
                if Q is None:
                    raise RuntimeError(
                        "Gram Newton-Schulz reset has no accumulated update"
                    )
                X = self.ops.matmul(Q, X)
                R = self.ops.symmetric_matmul(X, X.mT)
                Q = None

            Z = self.ops.symmetric_batch_matrix_matrix_product(
                R, R, C=R, alpha=c, beta=b
            )
            if Q is None:
                Q = Z + a * I
            else:
                Q = self.ops.symmetric_batch_matrix_matrix_product(Q, Z, C=Q, beta=a)
            if (
                iteration < len(self.coeff) - 1
                and iteration + 1 not in reset_iterations
            ):
                RZ = self.ops.symmetric_batch_matrix_matrix_product(R, Z, C=R, beta=a)
                R = self.ops.symmetric_batch_matrix_matrix_product(Z, RZ, C=RZ, beta=a)
        if Q is None:
            raise RuntimeError("Gram Newton-Schulz requires at least one coefficient")
        X = self.ops.matmul(Q, X)

        return X


class GramNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Sequence[Sequence[float]] = POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=[list(coefficients) for coefficients in ns_coefficients],
            use_gram=True,
            use_triton=ns_use_kernels,
            gns_reset_iters=list(gram_newton_schulz_reset_iterations),
            compile_kwargs=None,
        )


class StandardNewtonSchulz(NewtonSchulz):
    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Sequence[Sequence[float]] = POLAR_EXPRESS_COEFFICIENTS,
    ) -> None:
        super().__init__(
            eps=ns_epsilon,
            coeff=[list(coefficients) for coefficients in ns_coefficients],
            use_gram=False,
            use_triton=ns_use_kernels,
            compile_kwargs=None,
        )
