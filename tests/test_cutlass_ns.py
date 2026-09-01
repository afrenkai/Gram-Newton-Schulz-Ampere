from collections.abc import Generator

import pytest
import torch
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import (
    CutlassBackend,
    cutlass_baddbmm,
    cutlass_full_gemm_is_preferred,
    cutlass_has_supported_layout,
    cutlass_is_column_major,
    cutlass_is_installed,
    cutlass_supports,
    cutlass_symmetric_baddbmm,
    cutlass_symmetric_bmm,
    cutlass_symmetric_supports,
    select_cutlass_tactic,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        self.calls.append("symmetric_matmul")
        return left @ right

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor:
        self.calls.append("symmetric_batch_matrix_matrix_product")
        return torch.baddbmm(accumulator, left, right, alpha=alpha, beta=beta)

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        self.calls.append("matmul")
        return left @ right

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        self.calls.append("matmul_add")
        return torch.baddbmm(accumulator, left, right, beta=beta)


@pytest.fixture(scope="module")
def ampere_device() -> Generator[torch.device, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] != 8:
        pytest.skip("requires NVIDIA Ampere")
    if not cutlass_is_installed():
        pytest.skip("requires flashinfer-python")
    yield device


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 2e-1
    return 5e-3, 2e-2


def symmetric_random(
    batch_size: int,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        batch_size,
        dimension,
        dimension,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return ((source + source.mT) * 0.5).contiguous()


def test_layout_recognition_is_strict() -> None:
    row_major = torch.empty(2, 24, 16)
    column_major = torch.empty(2, 16, 24).mT
    irregular = torch.empty(2, 24, 32)[..., ::2]

    assert row_major.is_contiguous()
    assert not cutlass_is_column_major(row_major)
    assert cutlass_has_supported_layout(row_major)
    assert cutlass_is_column_major(column_major)
    assert cutlass_has_supported_layout(column_major)
    assert not cutlass_has_supported_layout(irregular)


@pytest.mark.parametrize(
    ("output_rows", "expected_tactic"),
    ((1, 1), (128, 1), (129, 2), (768, 2), (769, 0), (4096, 0)),
)
def test_tactic_boundaries(output_rows: int, expected_tactic: int) -> None:
    assert select_cutlass_tactic(output_rows) == expected_tactic


def test_full_gemm_policy_is_square_only() -> None:
    square_left = torch.empty(2, 64, 64)
    square_right = torch.empty(2, 64, 64)
    rectangular_left = torch.empty(2, 128, 64)

    assert cutlass_full_gemm_is_preferred(square_left, square_right)
    assert not cutlass_full_gemm_is_preferred(rectangular_left, square_right)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_symmetric_rr_matches_torch(
    ampere_device: torch.device,
    dtype: torch.dtype,
) -> None:
    left = symmetric_random(2, 136, dtype, ampere_device, 101)
    accumulator = symmetric_random(2, 136, dtype, ampere_device, 103)
    expected = torch.baddbmm(accumulator, left, left, alpha=0.75, beta=0.25)
    actual = cutlass_symmetric_baddbmm(
        accumulator,
        left,
        left,
        alpha=0.75,
        beta=0.25,
    )
    relative_tolerance, absolute_tolerance = tolerance(dtype)

    assert torch.equal(actual, actual.mT)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_symmetric_cr_gram_matches_torch(
    ampere_device: torch.device,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device=ampere_device).manual_seed(107)
    matrix = torch.randn(
        2,
        192,
        128,
        dtype=dtype,
        device=ampere_device,
        generator=generator,
    )
    left = matrix.mT
    accumulator = torch.zeros(2, 128, 128, dtype=dtype, device=ampere_device)
    expected = torch.baddbmm(accumulator, left, matrix, beta=0.0)
    actual = cutlass_symmetric_baddbmm(accumulator, left, matrix, beta=0.0)
    relative_tolerance, absolute_tolerance = tolerance(dtype)

    assert cutlass_is_column_major(left)
    assert torch.equal(actual, actual.mT)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )


@pytest.mark.parametrize("dimension", (40, 136, 264))
def test_symmetric_rc_partial_tiles_match_torch(
    ampere_device: torch.device,
    dimension: int,
) -> None:
    generator = torch.Generator(device=ampere_device).manual_seed(109 + dimension)
    matrix = torch.randn(
        1,
        dimension,
        72,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    right = matrix.mT
    expected = matrix @ right
    actual = cutlass_symmetric_bmm(matrix, right)

    assert cutlass_is_column_major(right)
    assert torch.equal(actual, actual.mT)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-2)


def test_symmetric_empty_beta_ignores_accumulator(
    ampere_device: torch.device,
) -> None:
    left = symmetric_random(1, 64, torch.float16, ampere_device, 127)
    accumulator = torch.full_like(left, torch.nan)
    actual = cutlass_symmetric_baddbmm(accumulator, left, left, beta=0.0)

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, actual.mT)


@pytest.mark.parametrize("tactic", (0, 1, 2))
def test_full_cutlass_tactics_match_torch(
    ampere_device: torch.device,
    tactic: int,
) -> None:
    generator = torch.Generator(device=ampere_device).manual_seed(131 + tactic)
    left = torch.randn(
        2,
        96,
        80,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    right_source = torch.randn(
        2,
        72,
        80,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    right = right_source.mT
    accumulator = torch.randn(
        2,
        96,
        72,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    expected = torch.baddbmm(accumulator, left, right, alpha=0.5, beta=0.25)
    actual = cutlass_baddbmm(
        accumulator,
        left,
        right,
        alpha=0.5,
        beta=0.25,
        tactic=tactic,
    )

    assert cutlass_supports(accumulator, left, right)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-2)


def test_direct_api_rejects_unsupported_inputs(
    ampere_device: torch.device,
) -> None:
    valid = torch.zeros(1, 64, 64, dtype=torch.float16, device=ampere_device)
    float32_tensor = valid.float()
    rank_two = valid[0]
    nonsquare_accumulator = torch.zeros(
        1,
        64,
        72,
        dtype=torch.float16,
        device=ampere_device,
    )
    both_column_major = valid.mT
    irregular = torch.empty(
        1,
        64,
        128,
        dtype=torch.float16,
        device=ampere_device,
    )[..., ::2]

    unsupported_cases = (
        (float32_tensor, float32_tensor, float32_tensor),
        (rank_two, rank_two, rank_two),
        (nonsquare_accumulator, valid, valid[..., :72]),
        (valid, both_column_major, both_column_major),
        (valid, irregular, valid),
    )
    for accumulator, left, right in unsupported_cases:
        with pytest.raises(ValueError, match="unsupported"):
            cutlass_symmetric_baddbmm(accumulator, left, right)


def test_direct_api_rejects_unaligned_and_unaligned_stride(
    ampere_device: torch.device,
) -> None:
    base = torch.empty(1 + 64 * 64, dtype=torch.float16, device=ampere_device)
    unaligned = torch.as_strided(
        base[1:],
        size=(1, 64, 64),
        stride=(64 * 64, 64, 1),
    )
    valid = torch.zeros(1, 64, 64, dtype=torch.float16, device=ampere_device)
    strided_base = torch.empty(
        2 * (64 * 64 + 4),
        dtype=torch.float16,
        device=ampere_device,
    )
    unaligned_batch_stride = torch.as_strided(
        strided_base,
        size=(2, 64, 64),
        stride=(64 * 64 + 4, 64, 1),
    )

    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16 != 0
    assert not cutlass_symmetric_supports(valid, unaligned, valid)
    assert not cutlass_symmetric_supports(
        unaligned_batch_stride,
        unaligned_batch_stride,
        unaligned_batch_stride,
    )
    with pytest.raises(ValueError, match="unsupported"):
        cutlass_symmetric_baddbmm(valid, unaligned, valid)


def test_invalid_full_tactic_is_rejected(ampere_device: torch.device) -> None:
    valid = torch.zeros(1, 64, 64, dtype=torch.float16, device=ampere_device)
    with pytest.raises(ValueError, match="tactic"):
        cutlass_baddbmm(valid, valid, valid, tactic=3)


def test_backend_falls_back_for_cpu_and_irregular_layout() -> None:
    fallback = RecordingBackend()
    backend = CutlassBackend(fallback)
    cpu_matrix = torch.randn(1, 16, 8)
    cpu_result = backend.symmetric_matmul(cpu_matrix, cpu_matrix.mT)

    assert fallback.calls == ["symmetric_matmul"]
    torch.testing.assert_close(cpu_result, cpu_matrix @ cpu_matrix.mT)


def test_backend_falls_back_for_unsupported_gpu_calls(
    ampere_device: torch.device,
) -> None:
    fallback = RecordingBackend()
    backend = CutlassBackend(fallback)
    source = torch.randn(1, 64, 128, device=ampere_device, dtype=torch.float16)
    irregular = source[..., ::2]
    symmetric_result = backend.symmetric_matmul(irregular, irregular.mT)
    rectangular_result = backend.matmul(source, source.new_empty(1, 128, 32))

    assert fallback.calls == ["symmetric_matmul", "matmul"]
    torch.testing.assert_close(symmetric_result, irregular @ irregular.mT)
    assert rectangular_result.shape == (1, 64, 32)


def test_false_symmetry_contract_reflects_upper_triangle(
    ampere_device: torch.device,
) -> None:
    generator = torch.Generator(device=ampere_device).manual_seed(149)
    left = torch.randn(
        1,
        64,
        64,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    right = torch.randn(
        1,
        64,
        64,
        dtype=torch.float16,
        device=ampere_device,
        generator=generator,
    )
    expected = left @ right
    actual = cutlass_symmetric_bmm(left, right)

    assert torch.equal(actual, actual.mT)
    torch.testing.assert_close(
        torch.triu(actual),
        torch.triu(expected),
        rtol=5e-3,
        atol=2e-2,
    )
    assert not torch.allclose(actual, expected, rtol=5e-3, atol=2e-2)
