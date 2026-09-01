#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from gram_newton_schulz import YOU_COEFFICIENTS  # ty: ignore[unresolved-import]
from gram_newton_schulz import (  # ty: ignore[unresolved-import]
    StandardNewtonSchulz as UpstreamStandardNewtonSchulz,
)
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import cutlass_symmetric_bmm
from gram_newton_schulz_ampere.newton_schulz import GramNewtonSchulz


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered_values = sorted(values)
    index = round(fraction * (len(ordered_values) - 1))
    return ordered_values[index]


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
    timings: list[float] = []
    completed_repeats = 0
    while completed_repeats < repeats:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        completed_repeats += 1
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


def parse_shape(value: str) -> tuple[int, int]:
    row_text, separator, column_text = value.lower().partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("shape must use ROWSxCOLUMNS")
    rows = int(row_text)
    columns = int(column_text)
    if rows <= 0 or columns <= 0 or rows % 8 != 0 or columns % 8 != 0:
        raise argparse.ArgumentTypeError(
            "shape dimensions must be positive multiples of 8"
        )
    return rows, columns


def relative_error(candidate: Tensor, reference: Tensor) -> float:
    difference = candidate.float() - reference.float()
    numerator = torch.linalg.vector_norm(difference)
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def orthogonality_residual(output: Tensor) -> float:
    if output.shape[-2] > output.shape[-1]:
        gram_matrix = output.mT @ output
    else:
        gram_matrix = output @ output.mT
    identity = torch.eye(
        gram_matrix.shape[-1],
        device=gram_matrix.device,
        dtype=gram_matrix.dtype,
    )
    error = gram_matrix.float() - identity.float()
    residuals = torch.linalg.vector_norm(error, dim=(-2, -1)) / math.sqrt(
        gram_matrix.shape[-1]
    )
    return float(residuals.mean().item())


def append_record(
    records: list[dict[str, float | int | str]],
    kind: str,
    name: str,
    batch_size: int,
    rows: int,
    columns: int,
    timing: dict[str, float],
    output: Tensor,
    reference: Tensor,
) -> None:
    smaller_dimension = min(rows, columns)
    larger_dimension = max(rows, columns)
    record: dict[str, float | int | str] = {
        "kind": kind,
        "name": name,
        "batch_size": batch_size,
        "rows": rows,
        "columns": columns,
        "smaller_dimension": smaller_dimension,
        "larger_dimension": larger_dimension,
        "aspect_ratio": larger_dimension / smaller_dimension,
        "orientation": (
            "tall" if rows > columns else "wide" if rows < columns else "square"
        ),
        "relative_error": relative_error(output, reference),
        **timing,
    }
    if kind == "end_to_end":
        record["orthogonality_residual"] = orthogonality_residual(output)
        record["finite"] = int(bool(torch.isfinite(output).all().item()))
    records.append(record)
    print(json.dumps(record), flush=True)


def benchmark_shape(
    batch_size: int,
    rows: int,
    columns: int,
    warmups: int,
    repeats: int,
    standard_operation: Callable[[Tensor], Tensor],
    torch_gram_operation: Callable[[Tensor], Tensor],
    cutlass_gram_operation: Callable[[Tensor], Tensor],
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device="cuda").manual_seed(
        67 + rows * 17 + columns * 31 + batch_size
    )
    benchmark_input = torch.randn(
        batch_size,
        rows,
        columns,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    primitive_input = benchmark_input.to(torch.float16)
    if rows > columns:
        left = primitive_input.mT
        right = primitive_input
    else:
        left = primitive_input
        right = primitive_input.mT
    torch_primitive_timing, torch_primitive = time_operation(
        lambda: left @ right,
        warmups,
        repeats,
    )
    append_record(
        records,
        "symmetric_gram",
        "torch",
        batch_size,
        rows,
        columns,
        torch_primitive_timing,
        torch_primitive,
        torch_primitive,
    )
    cutlass_primitive_timing, cutlass_primitive = time_operation(
        lambda: cutlass_symmetric_bmm(left, right),
        warmups,
        repeats,
    )
    append_record(
        records,
        "symmetric_gram",
        "cutlass_triangular",
        batch_size,
        rows,
        columns,
        cutlass_primitive_timing,
        cutlass_primitive,
        torch_primitive,
    )
    del primitive_input, torch_primitive, cutlass_primitive
    torch.cuda.empty_cache()

    torch_gram_timing, torch_gram_output = time_operation(
        lambda: torch_gram_operation(benchmark_input),
        warmups,
        repeats,
    )
    append_record(
        records,
        "end_to_end",
        "gram_torch_eager",
        batch_size,
        rows,
        columns,
        torch_gram_timing,
        torch_gram_output,
        torch_gram_output,
    )
    cutlass_gram_timing, cutlass_gram_output = time_operation(
        lambda: cutlass_gram_operation(benchmark_input),
        warmups,
        repeats,
    )
    append_record(
        records,
        "end_to_end",
        "gram_cutlass_triangular",
        batch_size,
        rows,
        columns,
        cutlass_gram_timing,
        cutlass_gram_output,
        torch_gram_output,
    )
    standard_timing, standard_output = time_operation(
        lambda: standard_operation(benchmark_input),
        warmups,
        repeats,
    )
    append_record(
        records,
        "end_to_end",
        "upstream_standard_torch_eager",
        batch_size,
        rows,
        columns,
        standard_timing,
        standard_output,
        torch_gram_output,
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=(
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (512, 256),
            (1024, 256),
            (2048, 256),
            (4096, 256),
            (4096, 512),
            (4096, 1024),
            (4096, 2048),
            (8192, 2048),
            (16384, 2048),
            (256, 512),
            (256, 1024),
            (256, 2048),
            (512, 4096),
            (1024, 8192),
            (2048, 16384),
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    standard_operation = UpstreamStandardNewtonSchulz(
        ns_epsilon=1e-7,
        ns_use_kernels=False,
        ns_coefficients=YOU_COEFFICIENTS,
        compile_kwargs=None,
    )
    torch_gram_operation = GramNewtonSchulz(
        ns_epsilon=1e-7,
        ns_backend="torch",
        ns_coefficients=YOU_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=(2,),
        ns_compile=False,
    )
    cutlass_gram_operation = GramNewtonSchulz(
        ns_epsilon=1e-7,
        ns_backend="cutlass",
        ns_coefficients=YOU_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=(2,),
        ns_compile=False,
    )
    records: list[dict[str, float | int | str]] = []
    for rows, columns in arguments.shapes:
        torch.cuda.empty_cache()
        records.extend(
            benchmark_shape(
                arguments.batch_size,
                rows,
                columns,
                arguments.warmups,
                arguments.repeats,
                standard_operation,
                torch_gram_operation,
                cutlass_gram_operation,
            )
        )
    payload: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "records": records,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
