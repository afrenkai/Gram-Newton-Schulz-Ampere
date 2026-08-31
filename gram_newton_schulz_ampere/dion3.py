from collections.abc import Mapping, Sequence
from math import ceil
from typing import cast

import torch
from torch import Tensor

from .coefficients import POLAR_EXPRESS_COEFFICIENTS
from .dion3_update import (
    apply_dion3_update,
    normalize_dion3_rows,
    select_dion3_rows,
)
from .muon import Muon
from .muon_distributed import (
    local_tensor,
    orthogonalize_parameter_updates,
    parameter_layout,
    resolve_gradient,
    validate_gradient_participation,
)
from .muon_types import (
    Coefficients,
    DistributedMesh,
    LearningRate,
    LearningRateAdjustment,
    LossClosure,
    NewtonSchulzAlgorithm,
    OptimizerAlgorithm,
    OptimizerParameters,
    ParameterGroup,
    ParameterGroupInput,
)


class Dion3(Muon):
    """Dion3 with Ampere Gram Newton--Schulz kernels."""

    def __init__(
        self,
        params: OptimizerParameters,
        lr: float = 0.01,
        fraction: float = 0.25,
        momentum: float = 0.95,
        muon_beta2: float = 0.95,
        mu: float | None = None,
        weight_decay: float = 0.01,
        *,
        distributed_mesh: DistributedMesh = None,
        betas: tuple[float, float] = (0.9, 0.95),
        epsilon: float = 1e-8,
        adjust_lr: LearningRateAdjustment = "spectral_norm",
        selection_scope: str = "local",
        ns_algorithm: NewtonSchulzAlgorithm = "gram_newton_schulz",
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Coefficients = POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        if mu is not None:
            if momentum != 0.95 and momentum != mu:
                raise ValueError("momentum and mu specify different values")
            momentum = mu
        parameter_groups = self.prepare_dion3_parameter_groups(params)
        super().__init__(
            parameter_groups,
            lr=lr,
            momentum=momentum,
            nesterov=False,
            weight_decay=weight_decay,
            distributed_mesh=distributed_mesh,
            betas=betas,
            epsilon=epsilon,
            adjust_lr=adjust_lr,
            flatten=False,
            ns_algorithm=ns_algorithm,
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            gram_newton_schulz_reset_iterations=(gram_newton_schulz_reset_iterations),
        )
        self.defaults["algorithm"] = "dion3"
        self.defaults["fraction"] = fraction
        self.defaults["muon_beta2"] = muon_beta2
        self.defaults["selection_scope"] = selection_scope
        for parameter_group in self.param_groups:
            if "fraction" not in parameter_group:
                parameter_group["fraction"] = fraction
            if "muon_beta2" not in parameter_group:
                parameter_group["muon_beta2"] = muon_beta2
            if "selection_scope" not in parameter_group:
                parameter_group["selection_scope"] = selection_scope
            self.validate_dion3_parameter_group(parameter_group)

    def accepts_dion3_parameter_groups(self) -> bool:
        return True

    def add_param_group(self, param_group: ParameterGroupInput) -> None:
        mutable_group = dict(param_group)
        if "algorithm" not in mutable_group:
            mutable_group["algorithm"] = "dion3"
        original_group_count = len(self.param_groups)
        original_state_parameters = set(self.state)
        try:
            super().add_param_group(mutable_group)
            if self.state_prepopulation_enabled:
                self.validate_dion3_parameter_group(self.param_groups[-1])
        except (TypeError, ValueError, RuntimeError):
            if len(self.param_groups) > original_group_count:
                self.param_groups.pop()
            added_state_parameters = set(self.state) - original_state_parameters
            for parameter in added_state_parameters:
                del self.state[parameter]
            raise

    @torch.no_grad()
    def step(self, closure: LossClosure | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for parameter_group in self.param_groups:
                algorithm = cast(OptimizerAlgorithm, parameter_group["algorithm"])
                if algorithm == "dion3":
                    self.step_dion3_group(parameter_group)
                elif algorithm == "adamw":
                    self.step_adamw_group(parameter_group)
                elif algorithm == "lion":
                    self.step_lion_group(parameter_group)
                else:
                    raise ValueError(f"Unknown Dion3 parameter algorithm: {algorithm}")
        return loss

    def prepare_dion3_parameter_groups(
        self,
        params: OptimizerParameters,
    ) -> tuple[ParameterGroup, ...]:
        parameter_inputs = list(params)
        if not parameter_inputs:
            raise ValueError("Dion3 requires at least one parameter or parameter group")
        if isinstance(parameter_inputs[0], Mapping):
            parameter_groups: list[ParameterGroup] = []
            for parameter_input in parameter_inputs:
                if not isinstance(parameter_input, Mapping):
                    raise TypeError("Parameter inputs cannot mix groups and tensors")
                parameter_group = dict(parameter_input)
                if "algorithm" not in parameter_group:
                    parameter_group["algorithm"] = "dion3"
                parameter_groups.append(parameter_group)
            return tuple(parameter_groups)
        return ({"params": parameter_inputs, "algorithm": "dion3"},)

    def validate_dion3_parameter_group(
        self,
        parameter_group: ParameterGroup,
    ) -> None:
        algorithm = parameter_group["algorithm"]
        if algorithm == "muon":
            raise ValueError("Muon parameter groups require the Muon optimizer")
        if algorithm != "dion3":
            return
        parameters = cast(list[Tensor], parameter_group["params"])
        invalid_shapes = [
            tuple(parameter.shape) for parameter in parameters if parameter.ndim < 2
        ]
        if invalid_shapes:
            raise ValueError(f"Dion3 requires matrix parameters, got {invalid_shapes}")
        fraction = parameter_group["fraction"]
        if not isinstance(fraction, float) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be a float in (0, 1], got {fraction}")
        muon_beta2 = parameter_group["muon_beta2"]
        if not isinstance(muon_beta2, float) or not 0.0 <= muon_beta2 < 1.0:
            raise ValueError(f"muon_beta2 must be a float in [0, 1), got {muon_beta2}")
        if parameter_group["selection_scope"] != "local":
            raise NotImplementedError(
                "Ampere Dion3 currently supports selection_scope='local'"
            )

    def step_dion3_group(self, parameter_group: ParameterGroup) -> None:
        all_parameters = cast(list[Tensor], parameter_group["params"])
        validate_gradient_participation(
            all_parameters,
            self.distributed_mesh,
            False,
        )
        parameters = [
            parameter for parameter in all_parameters if parameter.grad is not None
        ]
        if not parameters:
            return

        fraction = cast(float, parameter_group["fraction"])
        error_feedback_decay = cast(float, parameter_group["momentum"])
        selected_updates: list[Tensor] = []
        selected_indices: list[Tensor] = []
        selected_global_shapes: list[tuple[int, ...]] = []
        for parameter in parameters:
            layout = parameter_layout(parameter, self.distributed_mesh, False)
            matrix_row_dimension = parameter.ndim - 2
            if layout.sharded_tensor_dimension not in {None, matrix_row_dimension}:
                raise NotImplementedError(
                    "Dion3 supports replicated matrices and row-sharded DTensors"
                )
            selected_count = None
            if layout.sharded_tensor_dimension == matrix_row_dimension:
                if layout.process_group is None:
                    raise ValueError("A row-sharded parameter requires a process group")
                world_size = torch.distributed.get_world_size(layout.process_group)
                padded_local_rows = ceil(parameter.shape[-2] / world_size)
                selected_count = max(1, ceil(fraction * padded_local_rows))
                global_selected_rows = selected_count * world_size
            else:
                global_selected_rows = max(1, ceil(fraction * parameter.shape[-2]))

            gradient = resolve_gradient(parameter)
            self.reject_sparse_or_complex_gradient(gradient)
            parameter_state = self.state[parameter]
            momentum_buffer = local_tensor(
                cast(Tensor, parameter_state["momentum_buffer"])
            )
            selected_update, indices = select_dion3_rows(
                momentum_buffer,
                gradient,
                fraction,
                error_feedback_decay,
                selected_count,
            )
            selected_updates.append(selected_update)
            selected_indices.append(indices)
            selected_global_shapes.append((global_selected_rows, parameter.shape[1]))

        orthogonalized_updates = orthogonalize_parameter_updates(
            parameters,
            selected_updates,
            selected_global_shapes,
            self.newton_schulz,
            self.distributed_mesh,
            False,
        )
        learning_rate = cast(LearningRate, parameter_group["lr"])
        weight_decay = cast(LearningRate, parameter_group["weight_decay"])
        adjustment = cast(LearningRateAdjustment, parameter_group["adjust_lr"])
        muon_beta2 = cast(float, parameter_group["muon_beta2"])
        for parameter_index, parameter in enumerate(parameters):
            parameter_state = self.state[parameter]
            variance_neuron = local_tensor(
                cast(Tensor, parameter_state["variance_neuron"])
            )
            normalized_update = normalize_dion3_rows(
                orthogonalized_updates[parameter_index],
                variance_neuron,
                selected_indices[parameter_index],
                muon_beta2,
                1e-8,
            )
            adjusted_learning_rate = self.adjusted_learning_rate(
                learning_rate,
                tuple(parameter.shape),
                adjustment,
            )
            apply_dion3_update(
                local_tensor(parameter),
                normalized_update,
                selected_indices[parameter_index],
                learning_rate,
                weight_decay,
                adjusted_learning_rate,
            )
