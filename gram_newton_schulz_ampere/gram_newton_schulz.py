from collections.abc import Sequence

from newton_schulz import NewtonSchulz

from .coefficients import POLAR_EXPRESS_COEFFICIENTS


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
