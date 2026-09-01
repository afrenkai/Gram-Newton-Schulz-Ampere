#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from gram_newton_schulz import YOU_COEFFICIENTS  # ty: ignore[unresolved-import]
from gram_newton_schulz import (  # ty: ignore[unresolved-import]
    StandardNewtonSchulz as UpstreamStandardNewtonSchulz,
)
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import cutlass_symmetric_baddbmm
from gram_newton_schulz_ampere.newton_schulz import GramNewtonSchulz


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered_values = sorted(values)
    index = round(fraction * (len(ordered_values) - 1))
    return ordered_values[index]


def summarize_samples(prefix: str, values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_p20": percentile(values, 0.2),
        f"{prefix}_p80": percentile(values, 0.8),
        f"{prefix}_minimum": min(values),
        f"{prefix}_maximum": max(values),
    }


def parse_samples(path: Path) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "power_watts": [],
        "gpu_utilization_percent": [],
        "memory_utilization_percent": [],
        "graphics_clock_mhz": [],
        "memory_clock_mhz": [],
        "temperature_celsius": [],
    }
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != len(samples):
            continue
        for field_index, name in enumerate(samples):
            try:
                samples[name].append(float(fields[field_index]))
            except ValueError:
                continue
    return samples


def start_sampler(path: Path, interval_ms: int) -> subprocess.Popen[str]:
    query = ",".join(
        (
            "power.draw",
            "utilization.gpu",
            "utilization.memory",
            "clocks.current.graphics",
            "clocks.current.memory",
            "temperature.gpu",
        )
    )
    return subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "--loop-ms",
            str(interval_ms),
            "--filename",
            str(path),
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def measure_operation(
    name: str,
    operation: Callable[[], Tensor] | None,
    warmups: int,
    duration_seconds: float,
    sample_interval_ms: int,
    output_directory: Path,
) -> tuple[dict[str, float | int | str], Tensor | None]:
    output: Tensor | None = None
    if operation is not None:
        completed_warmups = 0
        while completed_warmups < warmups:
            output = operation()
            completed_warmups += 1
        torch.cuda.synchronize()
    sample_path = output_directory / f"{name}.csv"
    sampler = start_sampler(sample_path, sample_interval_ms)
    time.sleep(sample_interval_ms / 1000.0 * 2.0)
    measurement_start = time.perf_counter()
    completed_iterations = 0
    if operation is None:
        time.sleep(duration_seconds)
    else:
        while time.perf_counter() - measurement_start < duration_seconds:
            output = operation()
            torch.cuda.synchronize()
            completed_iterations += 1
    elapsed_seconds = time.perf_counter() - measurement_start
    sampler.terminate()
    sampler.wait(timeout=10)
    sample_values = parse_samples(sample_path)
    for sample_name in sample_values:
        sample_values[sample_name] = sample_values[sample_name][2:]
    record: dict[str, float | int | str] = {
        "kind": "power_utilization",
        "name": name,
        "elapsed_seconds": elapsed_seconds,
        "iterations": completed_iterations,
        "sample_count": len(sample_values["power_watts"]),
    }
    if completed_iterations > 0:
        record["iterations_per_second"] = completed_iterations / elapsed_seconds
        record["mean_iteration_ms"] = elapsed_seconds * 1000.0 / completed_iterations
    for sample_name, values in sample_values.items():
        record.update(summarize_samples(sample_name, values))
    if sample_values["power_watts"]:
        estimated_energy_joules = (
            statistics.fmean(sample_values["power_watts"]) * elapsed_seconds
        )
        record["estimated_energy_joules"] = estimated_energy_joules
        if completed_iterations > 0:
            record["estimated_joules_per_iteration"] = (
                estimated_energy_joules / completed_iterations
            )
    print(json.dumps(record), flush=True)
    return record, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--sample-interval-ms", type=int, default=50)
    parser.add_argument("--sample-directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    temporary_context = (
        tempfile.TemporaryDirectory(prefix="gns-power-")
        if arguments.sample_directory is None
        else None
    )
    if temporary_context is not None:
        sample_directory = Path(temporary_context.name)
    else:
        sample_directory = arguments.sample_directory
        sample_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    idle_record, idle_output = measure_operation(
        "idle",
        None,
        0,
        arguments.idle_seconds,
        arguments.sample_interval_ms,
        sample_directory,
    )
    records.append(idle_record)
    if idle_output is not None:
        raise RuntimeError("idle measurement unexpectedly produced output")

    generator = torch.Generator(device="cuda").manual_seed(191)
    matrix = torch.randn(
        arguments.batch_size,
        arguments.rows,
        arguments.columns,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    accumulator = torch.empty(
        arguments.batch_size,
        arguments.columns,
        arguments.columns,
        dtype=torch.float16,
        device="cuda",
    )
    primitive_operations: tuple[tuple[str, Callable[[], Tensor]], ...] = (
        ("torch_symmetric_gram", lambda: matrix.mT @ matrix),
        (
            "cutlass_separate_mirror",
            lambda: cutlass_symmetric_baddbmm(
                accumulator,
                matrix.mT,
                matrix,
                beta=0.0,
            ),
        ),
    )
    for name, operation in primitive_operations:
        record, output = measure_operation(
            name,
            operation,
            arguments.warmups,
            arguments.duration_seconds,
            arguments.sample_interval_ms,
            sample_directory,
        )
        if output is None or not torch.isfinite(output).all():
            raise RuntimeError(f"{name} produced invalid output")
        records.append(record)

    benchmark_input = matrix.to(torch.bfloat16)
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
    standard_operation = UpstreamStandardNewtonSchulz(
        ns_epsilon=1e-7,
        ns_use_kernels=False,
        ns_coefficients=YOU_COEFFICIENTS,
        compile_kwargs=None,
    )
    end_to_end_operations: tuple[tuple[str, Callable[[], Tensor]], ...] = (
        ("gram_torch_eager", lambda: torch_gram_operation(benchmark_input)),
        (
            "gram_cutlass_triangular",
            lambda: cutlass_gram_operation(benchmark_input),
        ),
        (
            "upstream_standard_torch_eager",
            lambda: standard_operation(benchmark_input),
        ),
    )
    for name, operation in end_to_end_operations:
        record, output = measure_operation(
            name,
            operation,
            arguments.warmups,
            arguments.duration_seconds,
            arguments.sample_interval_ms,
            sample_directory,
        )
        if output is None or not torch.isfinite(output).all():
            raise RuntimeError(f"{name} produced invalid output")
        records.append(record)

    payload: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sample_interval_ms": arguments.sample_interval_ms,
        "records": records,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)
    if temporary_context is not None:
        temporary_context.cleanup()


if __name__ == "__main__":
    main()
