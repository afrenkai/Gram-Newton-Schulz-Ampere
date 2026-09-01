#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from gram_newton_schulz import (
    YOU_COEFFICIENTS,
)
from gram_newton_schulz import (
    StandardNewtonSchulz as UpstreamStandardNewtonSchulz,
)
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import (
    CutlassBackend,
    cutlass_baddbmm,
    cutlass_symmetric_baddbmm,
    cutlass_symmetric_bmm,
)
from gram_newton_schulz_ampere.kernels.torch_ns import TorchBackend
from gram_newton_schulz_ampere.newton_schulz import (
    GramNewtonSchulz,
    build_gram_newton_schulz_operation,
)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered_values = sorted(values)
    percentile_index = round(fraction * (len(ordered_values) - 1))
    return ordered_values[percentile_index]


def time_operation(
    operation: Callable[[], Tensor],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float], Tensor]:
    output = operation()
    completed_warmups = 0
    while completed_warmups < warmups:
        output = operation()
        completed_warmups += 1
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for index in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for index in range(repeats)]
    for repeat_index in range(repeats):
        start_events[repeat_index].record()
        output = operation()
        end_events[repeat_index].record()
    torch.cuda.synchronize()
    timings = [
        start_events[index].elapsed_time(end_events[index]) for index in range(repeats)
    ]
    return (
        {
            "median_ms": statistics.median(timings),
            "p20_ms": percentile(timings, 0.2),
            "p80_ms": percentile(timings, 0.8),
            "minimum_ms": min(timings),
            "maximum_ms": max(timings),
        },
        output,
    )


def relative_error(candidate: Tensor, reference: Tensor) -> float:
    difference_norm_squared = torch.zeros((), device=candidate.device)
    reference_norm_squared = torch.zeros((), device=reference.device)
    for batch_index in range(candidate.shape[0]):
        difference = candidate[batch_index].float()
        difference.sub_(reference[batch_index])
        difference_norm_squared += torch.linalg.vector_norm(difference).square()
        reference_norm_squared += torch.linalg.vector_norm(
            reference[batch_index].float()
        ).square()
    return float(
        difference_norm_squared.sqrt() / reference_norm_squared.sqrt().clamp_min(1e-30)
    )


def maximum_asymmetry(matrix: Tensor) -> float:
    return float((matrix.float() - matrix.mT.float()).abs().max())


def record_timing(
    records: list[dict[str, float | int | str]],
    kind: str,
    name: str,
    batch_size: int,
    timing: dict[str, float],
    output: Tensor,
    reference: Tensor,
) -> None:
    record: dict[str, float | int | str] = {
        "kind": kind,
        "name": name,
        "batch_size": batch_size,
        "relative_error": relative_error(output, reference),
        "maximum_asymmetry": (
            maximum_asymmetry(output) if output.shape[-2] == output.shape[-1] else 0.0
        ),
        **timing,
    }
    records.append(record)
    print(json.dumps(record), flush=True)


def output_metrics(output: Tensor) -> tuple[bool, float, float]:
    finite = bool(torch.isfinite(output).all().item())
    output_rms = float(output.float().square().mean().sqrt().item())
    gram_matrix = output.mT @ output
    identity = torch.eye(
        gram_matrix.shape[-1],
        device=output.device,
        dtype=gram_matrix.dtype,
    )
    error = gram_matrix.float() - identity.float()
    batch_residuals = torch.linalg.vector_norm(error, dim=(-2, -1)) / math.sqrt(
        gram_matrix.shape[-1]
    )
    return finite, output_rms, float(batch_residuals.mean().item())


def record_end_to_end(
    records: list[dict[str, float | int | str]],
    name: str,
    batch_size: int,
    timing: dict[str, float],
    output: Tensor,
    reference: Tensor,
) -> None:
    finite, output_rms, orthogonality_residual = output_metrics(output)
    record: dict[str, float | int | str] = {
        "kind": "full_gns_end_to_end",
        "name": name,
        "batch_size": batch_size,
        "relative_error": relative_error(output, reference),
        "finite": int(finite),
        "output_rms": output_rms,
        "orthogonality_residual": orthogonality_residual,
        **timing,
    }
    records.append(record)
    print(json.dumps(record), flush=True)


def benchmark_batch(
    batch_size: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device="cuda").manual_seed(67)
    square_source = torch.randn(
        batch_size,
        2048,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    square_left = (square_source + square_source.mT).mul(0.5).contiguous()
    square_accumulator = square_left.clone()
    torch_square_timing, torch_square = time_operation(
        lambda: torch.baddbmm(
            square_accumulator,
            square_left,
            square_left,
            alpha=0.5,
            beta=0.75,
        ),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "symmetric_square_baddbmm",
        "torch",
        batch_size,
        torch_square_timing,
        torch_square,
        torch_square,
    )
    full_square_timing, full_square = time_operation(
        lambda: cutlass_baddbmm(
            square_accumulator,
            square_left,
            square_left,
            alpha=0.5,
            beta=0.75,
            tactic=0,
        ),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "symmetric_square_baddbmm",
        "cutlass_full",
        batch_size,
        full_square_timing,
        full_square,
        torch_square,
    )
    symmetric_square_timing, symmetric_square = time_operation(
        lambda: cutlass_symmetric_baddbmm(
            square_accumulator,
            square_left,
            square_left,
            alpha=0.5,
            beta=0.75,
        ),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "symmetric_square_baddbmm",
        "cutlass_triangular",
        batch_size,
        symmetric_square_timing,
        symmetric_square,
        torch_square,
    )
    torch.cuda.empty_cache()

    matrix = torch.randn(
        batch_size,
        16384,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    torch_gram_timing, torch_gram = time_operation(
        lambda: matrix.mT @ matrix,
        warmups,
        repeats,
    )
    record_timing(
        records,
        "symmetric_gram",
        "torch",
        batch_size,
        torch_gram_timing,
        torch_gram,
        torch_gram,
    )
    symmetric_gram_timing, symmetric_gram = time_operation(
        lambda: cutlass_symmetric_bmm(matrix.mT, matrix),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "symmetric_gram",
        "cutlass_triangular",
        batch_size,
        symmetric_gram_timing,
        symmetric_gram,
        torch_gram,
    )

    matrix = matrix / matrix.float().square().sum(dim=(-2, -1), keepdim=True).sqrt()
    matrix = matrix.to(torch.float16)
    torch_backend = TorchBackend()
    cutlass_backend = CutlassBackend(fallback=torch_backend)
    torch_operation = build_gram_newton_schulz_operation(
        backend=torch_backend,
        coefficients=YOU_COEFFICIENTS,
        reset_iterations=(2,),
        epsilon=1e-7,
        compile_operation=False,
        dynamic_compile=False,
        normalize_input=False,
    )
    cutlass_operation = build_gram_newton_schulz_operation(
        backend=cutlass_backend,
        coefficients=YOU_COEFFICIENTS,
        reset_iterations=(2,),
        epsilon=1e-7,
        compile_operation=False,
        dynamic_compile=False,
        normalize_input=False,
    )
    torch_core_timing, torch_core = time_operation(
        lambda: torch_operation(matrix),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "full_gns_core",
        "torch",
        batch_size,
        torch_core_timing,
        torch_core,
        torch_core,
    )
    cutlass_core_timing, cutlass_core = time_operation(
        lambda: cutlass_operation(matrix),
        warmups,
        repeats,
    )
    record_timing(
        records,
        "full_gns_core",
        "cutlass_triangular",
        batch_size,
        cutlass_core_timing,
        cutlass_core,
        torch_core,
    )
    del torch_core, cutlass_core
    matrix = torch.empty(0, device="cuda")
    torch.cuda.empty_cache()
    input_generator = torch.Generator(device="cuda").manual_seed(67)
    benchmark_input = torch.randn(
        batch_size,
        16384,
        2048,
        generator=input_generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    standard_end_to_end = UpstreamStandardNewtonSchulz(
        ns_epsilon=1e-7,
        ns_use_kernels=False,
        ns_coefficients=YOU_COEFFICIENTS,
        compile_kwargs=None,
    )
    torch_gram_end_to_end = GramNewtonSchulz(
        ns_epsilon=1e-7,
        ns_backend="torch",
        ns_coefficients=YOU_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=(2,),
        ns_compile=False,
    )
    cutlass_gram_end_to_end = GramNewtonSchulz(
        ns_epsilon=1e-7,
        ns_backend="cutlass",
        ns_coefficients=YOU_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=(2,),
        ns_compile=False,
    )
    standard_timing, standard_output = time_operation(
        lambda: standard_end_to_end(benchmark_input),
        warmups,
        repeats,
    )
    torch_gram_timing, torch_gram_output = time_operation(
        lambda: torch_gram_end_to_end(benchmark_input),
        warmups,
        repeats,
    )
    record_end_to_end(
        records,
        "upstream_standard_torch_eager",
        batch_size,
        standard_timing,
        standard_output,
        torch_gram_output,
    )
    standard_output = torch.empty(0, device="cuda")
    torch.cuda.empty_cache()
    record_end_to_end(
        records,
        "gram_torch_eager",
        batch_size,
        torch_gram_timing,
        torch_gram_output,
        torch_gram_output,
    )
    cutlass_gram_timing, cutlass_gram_output = time_operation(
        lambda: cutlass_gram_end_to_end(benchmark_input),
        warmups,
        repeats,
    )
    record_end_to_end(
        records,
        "gram_cutlass_triangular",
        batch_size,
        cutlass_gram_timing,
        cutlass_gram_output,
        torch_gram_output,
    )

    return records


def smoke_test() -> None:
    generator = torch.Generator(device="cuda").manual_seed(67)
    source = torch.randn(
        2,
        256,
        256,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    symmetric = (source + source.mT).mul(0.5).contiguous()
    reference = torch.baddbmm(symmetric, symmetric, symmetric, alpha=0.5, beta=0.75)
    candidate = cutlass_symmetric_baddbmm(
        symmetric,
        symmetric,
        symmetric,
        alpha=0.5,
        beta=0.75,
    )
    matrix = torch.randn(
        2,
        1024,
        256,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    gram_reference = matrix.mT @ matrix
    gram_candidate = cutlass_symmetric_bmm(matrix.mT, matrix)
    result = {
        "device": torch.cuda.get_device_name(0),
        "square_relative_error": relative_error(candidate, reference),
        "square_maximum_asymmetry": maximum_asymmetry(candidate),
        "gram_relative_error": relative_error(gram_candidate, gram_reference),
        "gram_maximum_asymmetry": maximum_asymmetry(gram_candidate),
    }
    print(json.dumps(result), flush=True)
    if not torch.isfinite(candidate).all() or not torch.isfinite(gram_candidate).all():
        raise RuntimeError("Symmetric CUTLASS smoke test produced non-finite output")
    if result["square_relative_error"] > 0.01 or result["gram_relative_error"] > 0.01:
        raise RuntimeError("Symmetric CUTLASS smoke test exceeded the accuracy gate")
    if result["square_maximum_asymmetry"] != 0.0:
        raise RuntimeError("Square CUTLASS output is not exactly symmetric")
    if result["gram_maximum_asymmetry"] != 0.0:
        raise RuntimeError("Gram CUTLASS output is not exactly symmetric")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 32))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.smoke:
        smoke_test()
        return
    records: list[dict[str, float | int | str]] = []
    for batch_size in arguments.batch_sizes:
        torch.cuda.empty_cache()
        records.extend(
            benchmark_batch(batch_size, arguments.warmups, arguments.repeats)
        )
    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "records": records,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
