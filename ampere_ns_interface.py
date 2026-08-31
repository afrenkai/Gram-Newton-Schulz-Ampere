from types import SimpleNamespace
import torch

try:
    from kernels import triton_ns as triton_kernels
except ImportError:
    triton_kernels = None


TORCH_BE = SimpleNamespace(
    symmetric_matmul=lambda A, B: A @ B,
    symmetric_batch_matrix_matrix_product=lambda A, B, C, alpha=1, beta=1: torch.baddbmm(
        C, A, B, alpha=alpha, beta=beta
    ),
    matmul=lambda A, B: A @ B,
    matmul_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)

TRITON_BE = (
    SimpleNamespace(
        symmetric_matmul=lambda A, B: A @ B,
        symmetric_batch_matrix_matrix_product=lambda A, B, C, alpha=1, beta=1: triton_kernels.triton_baddbmm(
            C, A, B, alpha=alpha, beta=beta
        ),
        matmul=lambda A, B: A @ B,
        matmul_add=lambda A, B, C, beta: triton_kernels.triton_baddbmm(C, A, B, beta=beta),
    )
    if triton_kernels is not None
    else None
)

