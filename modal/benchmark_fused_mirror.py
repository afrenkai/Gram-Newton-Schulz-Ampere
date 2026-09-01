#!/usr/bin/env python3
"""Benchmark the rejected prototype after applying fused_mirror_prototype.patch."""

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import (
    cutlass_symmetric_baddbmm,
    cutlass_symmetric_baddbmm_fused_mirror,
)


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


def relative_error(candidate: Tensor, reference: Tensor) -> float:
    difference = candidate.float() - reference.float()
    numerator = torch.linalg.vector_norm(difference)
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def make_operands(
    layout: str,
    batch_size: int,
    dimension: int,
    inner_dimension: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    if layout == "rr":
        source = torch.randn(
            batch_size,
            dimension,
            dimension,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        left = ((source + source.mT) * 0.5).contiguous()
        right = left
    elif layout == "cr":
        matrix = torch.randn(
            batch_size,
            inner_dimension,
            dimension,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        left = matrix.mT
        right = matrix
    elif layout == "rc":
        matrix = torch.randn(
            batch_size,
            dimension,
            inner_dimension,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        left = matrix
        right = matrix.mT
    else:
        raise ValueError(f"unknown layout {layout}")
    accumulator_source = torch.randn(
        batch_size,
        dimension,
        dimension,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    accumulator = ((accumulator_source + accumulator_source.mT) * 0.5).contiguous()
    return accumulator, left, right


def append_record(
    records: list[dict[str, float | int | str]],
    name: str,
    layout: str,
    batch_size: int,
    dimension: int,
    inner_dimension: int,
    timing: dict[str, float],
    output: Tensor,
    reference: Tensor,
    separate_reference: Tensor,
) -> None:
    record: dict[str, float | int | str] = {
        "kind": "fused_mirror",
        "name": name,
        "layout": layout,
        "batch_size": batch_size,
        "dimension": dimension,
        "inner_dimension": inner_dimension,
        "inner_to_output_ratio": inner_dimension / dimension,
        "relative_error": relative_error(output, reference),
        "maximum_asymmetry": float((output.float() - output.mT.float()).abs().max()),
        "bitwise_equal_to_separate": int(torch.equal(output, separate_reference)),
        **timing,
    }
    records.append(record)
    print(json.dumps(record), flush=True)


def benchmark_problem(
    records: list[dict[str, float | int | str]],
    layout: str,
    batch_size: int,
    dimension: int,
    inner_dimension: int,
    warmups: int,
    repeats: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        79 + batch_size * 13 + dimension * 17 + inner_dimension * 19
    )
    accumulator, left, right = make_operands(
        layout,
        batch_size,
        dimension,
        inner_dimension,
        generator,
    )
    torch_timing, torch_output = time_operation(
        lambda: torch.baddbmm(
            accumulator,
            left,
            right,
            alpha=0.75,
            beta=0.25,
        ),
        warmups,
        repeats,
    )
    separate_timing, separate_output = time_operation(
        lambda: cutlass_symmetric_baddbmm(
            accumulator,
            left,
            right,
            alpha=0.75,
            beta=0.25,
        ),
        warmups,
        repeats,
    )
    fused_timing, fused_output = time_operation(
        lambda: cutlass_symmetric_baddbmm_fused_mirror(
            accumulator,
            left,
            right,
            alpha=0.75,
            beta=0.25,
        ),
        warmups,
        repeats,
    )
    append_record(
        records,
        "torch_full",
        layout,
        batch_size,
        dimension,
        inner_dimension,
        torch_timing,
        torch_output,
        torch_output,
        separate_output,
    )
    append_record(
        records,
        "cutlass_separate_mirror",
        layout,
        batch_size,
        dimension,
        inner_dimension,
        separate_timing,
        separate_output,
        torch_output,
        separate_output,
    )
    append_record(
        records,
        "cutlass_fused_mirror",
        layout,
        batch_size,
        dimension,
        inner_dimension,
        fused_timing,
        fused_output,
        torch_output,
        separate_output,
    )
    if not torch.equal(fused_output, separate_output):
        raise RuntimeError("fused mirror output differs from the separate kernel")
    del torch_output, separate_output, fused_output
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=(256, 512, 1024, 2048),
    )
    parser.add_argument(
        "--inner-ratios",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    records: list[dict[str, float | int | str]] = []
    for batch_size in arguments.batch_sizes:
        for dimension in arguments.dimensions:
            benchmark_problem(
                records,
                "rr",
                batch_size,
                dimension,
                dimension,
                arguments.warmups,
                arguments.repeats,
            )
            for inner_ratio in arguments.inner_ratios:
                inner_dimension = dimension * inner_ratio
                benchmark_problem(
                    records,
                    "cr",
                    batch_size,
                    dimension,
                    inner_dimension,
                    arguments.warmups,
                    arguments.repeats,
                )
                benchmark_problem(
                    records,
                    "rc",
                    batch_size,
                    dimension,
                    inner_dimension,
                    arguments.warmups,
                    arguments.repeats,
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
