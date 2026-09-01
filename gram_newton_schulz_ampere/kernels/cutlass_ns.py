from collections.abc import Callable
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from gram_newton_schulz_ampere.kernels.protocols import (
    CutlassJitModule,
    FlashInferJitSpec,
    MatrixBackend,
)

try:
    from flashinfer.jit import gen_jit_spec  # ty: ignore[unresolved-import]
except ImportError:
    gen_jit_spec: Callable[..., object] | None = None


@cache
def load_cutlass_module() -> CutlassJitModule:
    if gen_jit_spec is None:
        raise RuntimeError(
            "The CUTLASS backend requires flashinfer-python with JIT support"
        )
    source_path = Path(__file__).with_suffix(".cu")
    source_digest = sha256(source_path.read_bytes()).hexdigest()[:12]
    specification = cast(
        FlashInferJitSpec,
        gen_jit_spec(
            f"gram_newton_schulz_ampere_sm80_{source_digest}",
            (source_path,),
            extra_cuda_cflags=[
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_80,code=compute_80",
            ],
        ),
    )
    module = specification.build_and_load()
    return cast(CutlassJitModule, module)


def cutlass_is_installed() -> bool:
    return gen_jit_spec is not None


def cutlass_is_column_major(tensor: Tensor) -> bool:
    return tensor.stride(-2) == 1 and tensor.stride(-1) == tensor.shape[-2]


def cutlass_has_supported_layout(tensor: Tensor) -> bool:
    return tensor.is_contiguous() or cutlass_is_column_major(tensor)


def cutlass_supports_problem(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
) -> bool:
    return (
        left.is_cuda
        and torch.cuda.get_device_capability(left.device)[0] == 8
        and left.dtype in {torch.float16, torch.bfloat16}
        and accumulator.dtype == left.dtype == right.dtype
        and accumulator.device == left.device == right.device
        and accumulator.ndim == left.ndim == right.ndim == 3
        and left.shape[0] == right.shape[0] == accumulator.shape[0]
        and left.shape[2] == right.shape[1]
        and left.shape[1] == accumulator.shape[1]
        and right.shape[2] == accumulator.shape[2]
        and accumulator.is_contiguous()
        and left.shape[2] % 8 == 0
        and right.shape[2] % 8 == 0
        and not (
            accumulator.data_ptr() % 16 or left.data_ptr() % 16 or right.data_ptr() % 16
        )
        and not (accumulator.stride(0) % 8 or left.stride(0) % 8 or right.stride(0) % 8)
    )


def cutlass_supports(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
) -> bool:
    return (
        cutlass_supports_problem(accumulator, left, right)
        and left.is_contiguous()
        and cutlass_has_supported_layout(right)
    )


def cutlass_symmetric_supports(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
) -> bool:
    return (
        cutlass_supports_problem(accumulator, left, right)
        and accumulator.shape[1] == accumulator.shape[2]
        and cutlass_has_supported_layout(left)
        and cutlass_has_supported_layout(right)
        and not (cutlass_is_column_major(left) and cutlass_is_column_major(right))
    )


def select_cutlass_tactic(rows: int, columns: int) -> int:
    if rows <= 128:
        return 1
    if rows <= 768:
        return 2
    return 0


def cutlass_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    tactic: int | None = None,
) -> Tensor:
    if not cutlass_supports(accumulator, left, right):
        raise ValueError(
            "Tensor shape, layout, dtype, or device is unsupported by SM80 CUTLASS"
        )
    selected_tactic = (
        select_cutlass_tactic(left.shape[-2], right.shape[-1])
        if tactic is None
        else tactic
    )
    if selected_tactic not in {0, 1, 2}:
        raise ValueError(f"CUTLASS tactic must be 0, 1, or 2, got {selected_tactic}")
    output = torch.empty_like(accumulator)
    right_column_major = not right.is_contiguous()
    load_cutlass_module().cutlass_baddbmm(
        accumulator,
        left,
        right,
        output,
        alpha,
        beta,
        selected_tactic,
        right_column_major,
    )
    return output


def cutlass_bmm(
    left: Tensor,
    right: Tensor,
    *,
    tactic: int | None = None,
) -> Tensor:
    output_shape = (*left.shape[:-2], left.shape[-2], right.shape[-1])
    accumulator = torch.empty(output_shape, dtype=left.dtype, device=left.device)
    return cutlass_baddbmm(
        accumulator,
        left,
        right,
        alpha=1.0,
        beta=0.0,
        tactic=tactic,
    )


def cutlass_symmetric_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    if not cutlass_symmetric_supports(accumulator, left, right):
        raise ValueError(
            "Tensor shape, layout, dtype, or device is unsupported by symmetric "
            "SM80 CUTLASS"
        )
    output = torch.empty_like(accumulator)
    load_cutlass_module().cutlass_symmetric_baddbmm(
        accumulator,
        left,
        right,
        output,
        alpha,
        beta,
        cutlass_is_column_major(left),
        cutlass_is_column_major(right),
    )
    return output


def cutlass_symmetric_bmm(left: Tensor, right: Tensor) -> Tensor:
    output_shape = (*left.shape[:-2], left.shape[-2], right.shape[-1])
    accumulator = torch.empty(output_shape, dtype=left.dtype, device=left.device)
    return cutlass_symmetric_baddbmm(
        accumulator,
        left,
        right,
        alpha=1.0,
        beta=0.0,
    )


def cutlass_problem_is_square(left: Tensor, right: Tensor) -> bool:
    return left.shape[-2] == left.shape[-1] and left.shape[-1] == right.shape[-1]


class CutlassBackend:
    def __init__(self, fallback: MatrixBackend) -> None:
        self.fallback = fallback

    def is_candidate(self, tensor: Tensor) -> bool:
        return (
            cutlass_is_installed()
            and tensor.is_cuda
            and torch.cuda.get_device_capability(tensor.device)[0] == 8
        )

    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        if not self.is_candidate(left):
            return self.fallback.symmetric_matmul(left, right)
        try:
            return cutlass_symmetric_bmm(left, right)
        except ValueError:
            return self.fallback.symmetric_matmul(left, right)

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
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
            return cutlass_symmetric_baddbmm(
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
        if not self.is_candidate(left) or not cutlass_problem_is_square(left, right):
            return self.fallback.matmul(left, right)
        try:
            return cutlass_bmm(left, right)
        except ValueError:
            return self.fallback.matmul(left, right)

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        if not self.is_candidate(left) or not cutlass_problem_is_square(left, right):
            return self.fallback.matmul_add(left, right, accumulator, beta)
        try:
            return cutlass_baddbmm(
                accumulator,
                left,
                right,
                beta=beta,
            )
        except ValueError:
            return self.fallback.matmul_add(left, right, accumulator, beta)
