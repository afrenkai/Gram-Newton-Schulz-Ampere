from functools import cache

from gram_newton_schulz_ampere.kernels.cutlass_ns import (
    CutlassBackend,
    cutlass_is_installed,
)
from gram_newton_schulz_ampere.kernels.protocols import MatrixBackend
from gram_newton_schulz_ampere.kernels.torch_ns import TorchBackend

try:
    from gram_newton_schulz_ampere.kernels import triton_ns
except ImportError:
    triton_ns = None


@cache
def select_matrix_backend(name: str) -> MatrixBackend:
    if name == "torch":
        return TorchBackend()
    if name == "cutlass":
        if not cutlass_is_installed():
            raise RuntimeError(
                "ns_backend='cutlass' requires flashinfer-python with JIT support"
            )
        return CutlassBackend(fallback=TorchBackend())
    if name == "triton":
        if triton_ns is None:
            raise RuntimeError("ns_backend='triton' requires Triton")
        return triton_ns.TritonBackend()
    raise ValueError(f"Unknown Newton--Schulz kernel backend: {name}")
