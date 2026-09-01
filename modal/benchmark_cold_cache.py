#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from gram_newton_schulz_ampere.kernels.cutlass_ns import (
    cutlass_symmetric_baddbmm,
)


def run_variant(
    variant: str,
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
) -> Tensor:
    if variant == "separate":
        return cutlass_symmetric_baddbmm(accumulator, left, right, beta=0.0)
    raise ValueError(f"unknown variant {variant}")


def worker(variant: str, dimension: int, inner_dimension: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(173)
    matrix = torch.randn(
        1,
        dimension,
        inner_dimension,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    accumulator = torch.empty(
        1,
        dimension,
        dimension,
        dtype=torch.float16,
        device="cuda",
    )
    torch.cuda.synchronize()
    first_call_start = time.perf_counter()
    output = run_variant(variant, accumulator, matrix, matrix.mT)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - first_call_start) * 1000.0

    warm_timings: list[float] = []
    completed_repeats = 0
    while completed_repeats < 20:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run_variant(variant, accumulator, matrix, matrix.mT)
        end.record()
        end.synchronize()
        warm_timings.append(start.elapsed_time(end))
        completed_repeats += 1
    cache_root = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    cache_files = [path for path in cache_root.rglob("*") if path.is_file()]
    result: dict[str, object] = {
        "kind": "worker",
        "variant": variant,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "dimension": dimension,
        "inner_dimension": inner_dimension,
        "first_call_ms": first_call_ms,
        "warm_median_ms": statistics.median(warm_timings),
        "cache_file_count": len(cache_files),
        "cache_bytes": sum(path.stat().st_size for path in cache_files),
        "finite": bool(torch.isfinite(output).all().item()),
    }
    print(json.dumps(result), flush=True)


def run_process(
    mode: str,
    trial: int,
    cache_path: Path,
    variant: str,
    dimension: int,
    inner_dimension: int,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["FLASHINFER_WORKSPACE_BASE"] = str(cache_path)
    environment["FLASHINFER_JIT_VERBOSE"] = "0"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        variant,
        "--dimension",
        str(dimension),
        "--inner-dimension",
        str(inner_dimension),
    ]
    process_start = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    process_wall_ms = (time.perf_counter() - process_start) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"cache trial failed with {completed.returncode}: {completed.stderr}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    worker_result = cast(dict[str, object], json.loads(output_lines[-1]))
    record: dict[str, object] = {
        **worker_result,
        "kind": mode,
        "trial": trial,
        "process_wall_ms": process_wall_ms,
        "cache_path": str(cache_path),
        "stderr": completed.stderr,
    }
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=("separate",), default="separate")
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--inner-dimension", type=int, default=16384)
    parser.add_argument("--cold-trials", type=int, default=2)
    parser.add_argument("--warm-process-trials", type=int, default=2)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.variant, arguments.dimension, arguments.inner_dimension)
        return

    records: list[dict[str, object]] = []
    temporary_context = (
        tempfile.TemporaryDirectory(prefix="gns-cutlass-cache-")
        if arguments.cache_root is None
        else None
    )
    if temporary_context is not None:
        cache_root = Path(temporary_context.name)
    else:
        cache_root = arguments.cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
    completed_cold_trials = 0
    while completed_cold_trials < arguments.cold_trials:
        cold_cache = cache_root / f"cold-{completed_cold_trials}"
        records.append(
            run_process(
                "cold_cache",
                completed_cold_trials,
                cold_cache,
                arguments.variant,
                arguments.dimension,
                arguments.inner_dimension,
            )
        )
        completed_cold_trials += 1
    reused_cache = cache_root / "cold-0"
    completed_warm_trials = 0
    while completed_warm_trials < arguments.warm_process_trials:
        records.append(
            run_process(
                "warm_process_cache",
                completed_warm_trials,
                reused_cache,
                arguments.variant,
                arguments.dimension,
                arguments.inner_dimension,
            )
        )
        completed_warm_trials += 1
    payload: dict[str, object] = {
        "python": sys.version,
        "records": records,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)
    if temporary_context is not None:
        temporary_context.cleanup()


if __name__ == "__main__":
    main()
