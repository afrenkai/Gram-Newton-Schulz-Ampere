# ruff: noqa: F722
from math import ceil

import torch
from beartype import beartype
from einops import repeat
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, distributed
from torch.distributed import ProcessGroup

from gram_newton_schulz_ampere.optimizer_types import LearningRate


@jaxtyped(typechecker=beartype)
def select_dion3_rows(
    momentum_buffer: Float[Tensor, "*batch rows columns"],
    gradient: Float[Tensor, "*batch rows columns"],
    fraction: float,
    error_feedback_decay: float,
    selected_count: int | None = None,
) -> tuple[
    Float[Tensor, "*batch selected columns"],
    Int[Tensor, "*indices"],
]:
    momentum_buffer.add_(gradient.to(dtype=momentum_buffer.dtype))
    row_count = momentum_buffer.shape[-2]
    if row_count == 0:
        indices = torch.empty(
            (*momentum_buffer.shape[:-2], 0),
            dtype=torch.long,
            device=momentum_buffer.device,
        )
        return momentum_buffer.to(dtype=torch.bfloat16), indices

    requested_count = selected_count
    if requested_count is None:
        requested_count = max(1, ceil(fraction * row_count))
    selected_count = min(requested_count, row_count)
    if selected_count == row_count:
        indices = torch.arange(row_count, device=momentum_buffer.device).expand(
            *momentum_buffer.shape[:-2],
            row_count,
        )
        selected_rows = momentum_buffer.clone()
        momentum_buffer.mul_(error_feedback_decay)
        return selected_rows.to(dtype=torch.bfloat16), indices

    row_norms = momentum_buffer.norm(p=1, dim=-1)
    indices = torch.topk(
        row_norms,
        selected_count,
        dim=-1,
        sorted=False,
    ).indices
    expanded_indices = repeat(
        indices,
        "... selected -> ... selected columns",
        columns=momentum_buffer.shape[-1],
    )
    selected_rows = torch.gather(
        momentum_buffer,
        dim=-2,
        index=expanded_indices,
    )
    momentum_buffer.scatter_(
        -2,
        expanded_indices,
        selected_rows * error_feedback_decay,
    )
    return selected_rows.to(dtype=torch.bfloat16), indices


@jaxtyped(typechecker=beartype)
def normalize_dion3_rows(
    selected_update: Float[Tensor, "*batch selected columns"],
    variance_neuron: Float[Tensor, "*batch rows 1"],
    indices: Int[Tensor, "*indices"],
    muon_beta2: float,
    epsilon: float,
    process_group: ProcessGroup | None = None,
) -> Float[Tensor, "*batch selected columns"]:
    variance_indices = repeat(
        indices,
        "... selected -> ... selected singleton",
        singleton=1,
    )
    selected_variance = torch.gather(
        variance_neuron,
        dim=-2,
        index=variance_indices,
    ).float()
    update = selected_update.to(dtype=torch.float32)
    original_squared_norm = update.square().sum(dim=(-2, -1), keepdim=True)
    if process_group is not None:
        distributed.all_reduce(
            original_squared_norm,
            op=distributed.ReduceOp.SUM,
            group=process_group,
        )
    original_norm = original_squared_norm.sqrt()
    neuron_variance = update.square().mean(dim=-1, keepdim=True)
    updated_variance = torch.lerp(
        selected_variance,
        neuron_variance,
        1 - muon_beta2,
    )
    normalized_update = update / (updated_variance.sqrt() + epsilon)
    normalized_squared_norm = normalized_update.square().sum(
        dim=(-2, -1),
        keepdim=True,
    )
    if process_group is not None:
        distributed.all_reduce(
            normalized_squared_norm,
            op=distributed.ReduceOp.SUM,
            group=process_group,
        )
    normalized_norm = normalized_squared_norm.sqrt().clamp(min=epsilon)
    normalized_update.mul_(original_norm / normalized_norm)
    variance_neuron.scatter_(
        -2,
        variance_indices,
        updated_variance.to(dtype=variance_neuron.dtype),
    )
    return normalized_update


@jaxtyped(typechecker=beartype)
def apply_dion3_update(
    parameter: Float[Tensor, "*batch rows columns"],
    selected_update: Float[Tensor, "*batch selected columns"],
    indices: Int[Tensor, "*indices"],
    learning_rate: LearningRate,
    weight_decay: LearningRate,
    adjusted_learning_rate: LearningRate,
) -> None:
    if not isinstance(weight_decay, float) or weight_decay != 0.0:
        parameter.mul_(1 - learning_rate * weight_decay)
    expanded_indices = repeat(
        indices,
        "... selected -> ... selected columns",
        columns=parameter.shape[-1],
    )
    parameter.scatter_add_(
        -2,
        expanded_indices,
        selected_update.to(dtype=parameter.dtype) * -adjusted_learning_rate,
    )
