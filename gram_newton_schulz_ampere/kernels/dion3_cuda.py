# ruff: noqa: F722
from collections.abc import Sequence
from functools import cache
from math import ceil
from pathlib import Path
from typing import cast

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.utils.cpp_extension import load

from gram_newton_schulz_ampere.kernels.types import Dion3CudaExtension


@cache
def dion3_cuda_extension() -> Dion3CudaExtension:
    source_path = Path(__file__).with_suffix(".cu")
    loaded_extension = load(
        name="gram_newton_schulz_ampere_dion3_cuda",
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        with_cuda=True,
        verbose=False,
    )
    return cast(Dion3CudaExtension, loaded_extension)


@jaxtyped(typechecker=beartype)
def momentum_row_norm_cuda(
    momentum: Float[Tensor, "*batch rows columns"],
    gradient: Float[Tensor, "*batch rows columns"],
) -> Float[Tensor, "*batch rows"]:
    return dion3_cuda_extension().momentum_row_norm(momentum, gradient)


@jaxtyped(typechecker=beartype)
def gather_decay_cuda(
    momentum: Float[Tensor, "*batch rows columns"],
    indices: Int[Tensor, "*batch selected"],
    decay: float,
) -> Float[Tensor, "*batch selected columns"]:
    return dion3_cuda_extension().gather_decay(momentum, indices, decay)


@jaxtyped(typechecker=beartype)
def select_dion3_rows_cuda(
    momentum: Float[Tensor, "*batch rows columns"],
    gradient: Float[Tensor, "*batch rows columns"],
    fraction: float,
    decay: float,
    selected_count: int | None = None,
    parameter: Float[Tensor, "*batch rows columns"] | None = None,
    parameter_decay: float = 1.0,
) -> tuple[
    Float[Tensor, "*batch selected columns"],
    Int[Tensor, "*batch selected"],
]:
    row_count = momentum.shape[-2]
    contiguous_gradient = gradient.to(dtype=momentum.dtype).contiguous()
    if row_count == 0:
        momentum.add_(contiguous_gradient)
        if parameter is not None and parameter_decay != 1.0:
            parameter.mul_(parameter_decay)
        indices = torch.empty(
            (*momentum.shape[:-2], 0),
            dtype=torch.long,
            device=momentum.device,
        )
        return momentum.to(dtype=torch.bfloat16), indices

    requested_count = selected_count
    if requested_count is None:
        requested_count = max(1, ceil(fraction * row_count))
    selected_count = min(requested_count, row_count)
    if selected_count == row_count:
        momentum.add_(contiguous_gradient)
        if parameter is not None and parameter_decay != 1.0:
            parameter.mul_(parameter_decay)
        indices = (
            torch.arange(row_count, device=momentum.device)
            .expand(*momentum.shape[:-2], row_count)
            .contiguous()
        )
        selected_rows = momentum.clone()
        momentum.mul_(decay)
        return selected_rows.to(dtype=torch.bfloat16), indices

    selection_parameter = momentum if parameter is None else parameter
    outputs = dion3_cuda_extension().select_rows(
        momentum,
        contiguous_gradient,
        selection_parameter,
        selected_count,
        decay,
        parameter_decay,
    )
    return outputs[0], outputs[1]


def select_dion3_rows_cuda_batch(
    momenta: Sequence[Tensor],
    gradients: Sequence[Tensor],
    parameters: Sequence[Tensor],
    selected_counts: Sequence[int],
    decay: float,
    parameter_decay: float,
) -> tuple[list[Tensor], list[Tensor]]:
    tensor_count = len(momenta)
    if not (
        len(gradients) == tensor_count
        and len(parameters) == tensor_count
        and len(selected_counts) == tensor_count
    ):
        raise ValueError("batched selection inputs must have equal lengths")
    if tensor_count == 0:
        return [], []
    contiguous_gradients = [
        gradients[tensor_index].to(dtype=momenta[tensor_index].dtype).contiguous()
        for tensor_index in range(tensor_count)
    ]
    for tensor_index in range(tensor_count):
        row_count = momenta[tensor_index].shape[-2]
        if not 0 < selected_counts[tensor_index] < row_count:
            raise ValueError("batched selection requires partial row selection")
    outputs = dion3_cuda_extension().select_rows_batch(
        list(momenta),
        contiguous_gradients,
        list(parameters),
        list(selected_counts),
        decay,
        parameter_decay,
    )
    selected_updates = [
        outputs[tensor_index * 2] for tensor_index in range(tensor_count)
    ]
    indices = [outputs[tensor_index * 2 + 1] for tensor_index in range(tensor_count)]
    return selected_updates, indices


@jaxtyped(typechecker=beartype)
def normalize_rows_cuda(
    selected_update: Float[Tensor, "*batch selected columns"],
    variance: Float[Tensor, "*batch rows 1"],
    indices: Int[Tensor, "*batch selected"],
    beta_two: float,
    epsilon: float,
) -> tuple[
    Float[Tensor, "*batch selected columns"],
    Float[Tensor, "batch 2"],
]:
    outputs = dion3_cuda_extension().normalize_rows(
        selected_update,
        variance,
        indices,
        beta_two,
        epsilon,
    )
    return outputs[0], outputs[1]


@jaxtyped(typechecker=beartype)
def apply_rows_cuda(
    parameter: Float[Tensor, "*batch rows columns"],
    normalized_update: Float[Tensor, "*batch selected columns"],
    indices: Int[Tensor, "*batch selected"],
    squared_norms: Float[Tensor, "batch 2"],
    learning_rate: float,
    weight_decay: float,
    adjusted_learning_rate: float,
    epsilon: float,
) -> None:
    dion3_cuda_extension().apply_rows(
        parameter,
        normalized_update,
        indices,
        squared_norms,
        learning_rate,
        weight_decay,
        adjusted_learning_rate,
        epsilon,
    )


@jaxtyped(typechecker=beartype)
def normalize_apply_rows_cuda(
    parameter: Float[Tensor, "*batch rows columns"],
    selected_update: Float[Tensor, "*batch selected columns"],
    variance: Float[Tensor, "*batch rows 1"],
    indices: Int[Tensor, "*batch selected"],
    beta_two: float,
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
    adjusted_learning_rate: float,
) -> None:
    dion3_cuda_extension().normalize_apply_rows(
        parameter,
        selected_update,
        variance,
        indices,
        beta_two,
        epsilon,
        learning_rate,
        weight_decay,
        adjusted_learning_rate,
    )


def normalize_apply_rows_cuda_batch(
    parameters: Sequence[Tensor],
    selected_updates: Sequence[Tensor],
    variances: Sequence[Tensor],
    indices: Sequence[Tensor],
    beta_two: float,
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
    adjusted_learning_rates: Sequence[float],
) -> None:
    tensor_count = len(parameters)
    if not (
        len(selected_updates) == tensor_count
        and len(variances) == tensor_count
        and len(indices) == tensor_count
        and len(adjusted_learning_rates) == tensor_count
    ):
        raise ValueError("batched normalization inputs must have equal lengths")
    contiguous_updates = [
        selected_updates[tensor_index].contiguous()
        for tensor_index in range(tensor_count)
    ]
    dion3_cuda_extension().normalize_apply_rows_batch(
        list(parameters),
        contiguous_updates,
        list(variances),
        list(indices),
        beta_two,
        epsilon,
        learning_rate,
        weight_decay,
        list(adjusted_learning_rates),
    )
