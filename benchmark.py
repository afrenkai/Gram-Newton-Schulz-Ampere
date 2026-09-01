# ruff: noqa: F722
import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from gram_newton_schulz_ampere.benchmark_schema import (
    BenchmarkConfig,
    MatrixOperation,
)
from gram_newton_schulz_ampere.newton_schulz import (
    GramNewtonSchulz,
    StandardNewtonSchulz,
)


def parse_shape(value: str) -> tuple[int, int]:
    row_text, separator, column_text = value.lower().partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("shape must use ROWSxCOLUMNS")
    rows, columns = int(row_text), int(column_text)
    if min(rows, columns) <= 0 or rows % 8 or columns % 8:
        raise argparse.ArgumentTypeError(
            "shape dimensions must be positive multiples of 8"
        )
    return rows, columns


def parse_coefficients(value: str) -> tuple[float, float, float]:
    values = value.split(",")
    if len(values) != 3:
        raise argparse.ArgumentTypeError("coefficient must use A,B,C")
    return float(values[0]), float(values[1]), float(values[2])


def parse_arguments() -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--batch-size", action="append", type=int, required=True)
    parser.add_argument(
        "--coefficient",
        action="append",
        type=parse_coefficients,
        required=True,
    )
    parser.add_argument("--reset-iteration", action="append", type=int)
    parser.add_argument(
        "--operation",
        action="append",
        choices=("standard", "torch-gns", "cutlass-gns"),
        required=True,
    )
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument(
        "--compile-operations",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if min(arguments.batch_size) <= 0:
        parser.error("batch sizes must be positive")
    if arguments.warmups < 0 or arguments.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats must be positive")
    return BenchmarkConfig(
        shapes=tuple(arguments.shape),
        batch_sizes=tuple(arguments.batch_size),
        coefficients=tuple(arguments.coefficient),
        reset_iterations=tuple(arguments.reset_iteration or ()),
        operations=tuple(arguments.operation),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        seed=arguments.seed,
        dtype=arguments.dtype,
        device=arguments.device,
        epsilon=arguments.epsilon,
        compile_operations=arguments.compile_operations,
        output=arguments.output,
    )


def select_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name}")


@jaxtyped(typechecker=beartype)
def time_operation(
    operation: MatrixOperation,
    matrix: Float[Tensor, "batch rows columns"],
    warmups: int,
    repeats: int,
) -> tuple[
    dict[str, float | list[float]],
    Float[Tensor, "batch rows columns"],
]:
    completed_warmups = 0
    while completed_warmups < warmups:
        operation(matrix)
        completed_warmups += 1
    torch.cuda.synchronize(matrix.device)
    timings: list[float] = []
    output = matrix
    completed_repeats = 0
    while completed_repeats < repeats:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation(matrix)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        completed_repeats += 1
    return {
        "median_ms": statistics.median(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "samples_ms": timings,
    }, output


@jaxtyped(typechecker=beartype)
def relative_error(
    candidate: Float[Tensor, "batch rows columns"],
    reference: Float[Tensor, "batch rows columns"],
) -> float:
    difference_norm = torch.linalg.vector_norm(candidate.float() - reference.float())
    reference_norm = torch.linalg.vector_norm(reference.float())
    return float(
        (
            difference_norm / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
        ).item()
    )


@jaxtyped(typechecker=beartype)
def orthogonality_residual(
    output: Float[Tensor, "batch rows columns"],
) -> float:
    gram = (
        output.mT @ output
        if output.shape[-2] > output.shape[-1]
        else output @ output.mT
    )
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    residual = torch.linalg.vector_norm(gram.float() - identity.float(), dim=(-2, -1))
    return float((residual / math.sqrt(gram.shape[-1])).mean().item())


def build_operations(
    config: BenchmarkConfig,
) -> tuple[dict[str, MatrixOperation], MatrixOperation]:
    torch_operation = GramNewtonSchulz(
        ns_epsilon=config.epsilon,
        ns_backend="torch",
        ns_coefficients=config.coefficients,
        gram_newton_schulz_reset_iterations=config.reset_iterations,
        ns_compile=config.compile_operations,
    )
    operations: dict[str, MatrixOperation] = {
        "standard": StandardNewtonSchulz(
            ns_epsilon=config.epsilon,
            ns_backend="torch",
            ns_coefficients=config.coefficients,
            ns_compile=config.compile_operations,
        ),
        "torch-gns": torch_operation,
        "cutlass-gns": GramNewtonSchulz(
            ns_epsilon=config.epsilon,
            ns_backend="cutlass",
            ns_coefficients=config.coefficients,
            ns_compile=config.compile_operations,
        ),
    }
    return operations, torch_operation


def main() -> None:
    config = parse_arguments()
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    operations, reference_operation = build_operations(config)
    records: list[dict[str, float | int | str | list[float]]] = []
    for rows, columns in config.shapes:
        for batch_size in config.batch_sizes:
            matrix = torch.randn(
                batch_size,
                rows,
                columns,
                generator=generator,
                device=device,
                dtype=select_dtype(config.dtype),
            )
            reference = reference_operation(matrix)
            for name in config.operations:
                timing, output = time_operation(
                    operations[name],
                    matrix,
                    config.warmups,
                    config.repeats,
                )
                records.append(
                    {
                        "name": name,
                        "batch_size": batch_size,
                        "rows": rows,
                        "columns": columns,
                        "finite": int(bool(torch.isfinite(output).all().item())),
                        "relative_error": relative_error(output, reference),
                        "orthogonality_residual": orthogonality_residual(output),
                        **timing,
                    }
                )
            torch.cuda.empty_cache()
    payload: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "records": records,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload) + "\n")
    print(config.output)


if __name__ == "__main__":
    main()
