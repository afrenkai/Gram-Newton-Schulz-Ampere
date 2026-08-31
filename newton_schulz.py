from types import SimpleNamespace

import torch

import triton_ns as triton_kernels

# https://arxiv.org/pdf/2505.16932
_unmodified_polar_express_coefficients = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
safety_factor = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / safety_factor, b / safety_factor**3, c / safety_factor**5)
    for (a, b, c) in _unmodified_polar_express_coefficients
]

_TORCH_BE = SimpleNamespace(
    symmetric_matmul=lambda A, B: A @ B,
    symmetric_batch_matrix_matrix_product=lambda A, B, C, alpha=1, beta=1: torch.baddbmm(
        C, A, B, alpha=alpha, beta=beta
    ),
    matmul=lambda A, B: A @ B,
    matmul_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)

_TRITON_BE = SimpleNamespace(
    symmetric_matmul=lambda A, B: A @ B,
    symmetric_batch_matrix_matrix_product=lambda A, B, C, alpha=1, beta=1: triton_kernels.triton_baddbmm(
        C, A, B, alpha=alpha, beta=beta
    ),
    matmul=lambda A, B: A @ B,
    matmul_add=lambda A, B, C, beta: triton_kernels.triton_baddbmm(C, A, B, beta=beta),
)


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
        self.ops = _TRITON_BE if use_triton else _TORCH_BE
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
