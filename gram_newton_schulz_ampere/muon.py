from collections.abc import Sequence
from math import sqrt
from typing import cast

import torch
from einops import rearrange
from torch import Tensor
from torch.optim.optimizer import Optimizer, StateDict

from .coefficients import POLAR_EXPRESS_COEFFICIENTS
from .muon_distributed import (
    local_tensor,
    orthogonalize_parameter_updates,
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
    NewtonSchulzBackend,
    OptimizerAlgorithm,
    OptimizerParameters,
    ParameterGroup,
    ParameterGroupInput,
)
from .newton_schulz import GramNewtonSchulz, NewtonSchulz, StandardNewtonSchulz


class Muon(Optimizer):
    """Muon with Ampere kernels and distributed PyTorch tensors."""

    def __init__(
        self,
        params: OptimizerParameters,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        *,
        distributed_mesh: DistributedMesh = None,
        betas: tuple[float, float] = (0.9, 0.95),
        epsilon: float = 1e-8,
        adjust_lr: LearningRateAdjustment = "spectral_norm",
        flatten: bool = False,
        ns_algorithm: NewtonSchulzAlgorithm = "auto",
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_backend: NewtonSchulzBackend = "auto",
        ns_coefficients: Coefficients = POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations: Sequence[int] = (2,),
    ) -> None:
        self.validate_constructor_values(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            epsilon=epsilon,
            ns_epsilon=ns_epsilon,
            ns_algorithm=ns_algorithm,
            ns_backend=ns_backend,
            adjust_lr=adjust_lr,
        )
        self.distributed_mesh = distributed_mesh
        self.ns_algorithm = ns_algorithm
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_backend = ns_backend
        self.ns_coefficients = tuple(
            tuple(float(coefficient) for coefficient in iteration_coefficients)
            for iteration_coefficients in ns_coefficients
        )
        self.gram_newton_schulz_reset_iterations = tuple(
            gram_newton_schulz_reset_iterations
        )
        self.state_prepopulation_enabled = False
        defaults: ParameterGroup = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "algorithm": "muon",
            "betas": betas,
            "epsilon": epsilon,
            "adjust_lr": adjust_lr,
            "flatten": flatten,
            "ns_algorithm": self.ns_algorithm,
            "ns_epsilon": self.ns_epsilon,
            "ns_use_kernels": self.ns_use_kernels,
            "ns_backend": self.ns_backend,
            "ns_coefficients": self.ns_coefficients,
            "gram_newton_schulz_reset_iterations": (
                self.gram_newton_schulz_reset_iterations
            ),
        }
        super().__init__(params, defaults)
        for parameter_group in self.param_groups:
            self.validate_parameter_group(parameter_group)
        self.validate_optimizer_devices()

        self.configure_newton_schulz()

        for parameter_group in self.param_groups:
            self.initialize_parameter_group_state(parameter_group)
        self.state_prepopulation_enabled = True

    def configure_newton_schulz(self) -> None:
        use_kernels = self.ns_use_kernels and self.has_cuda_parameters()
        self.newton_schulz = self.build_newton_schulz(
            ns_algorithm=self.ns_algorithm,
            ns_epsilon=self.ns_epsilon,
            ns_use_kernels=use_kernels,
            ns_backend=self.ns_backend,
            ns_coefficients=self.ns_coefficients,
            gram_newton_schulz_reset_iterations=(
                self.gram_newton_schulz_reset_iterations
            ),
        )

    def load_state_dict(self, state_dict: StateDict) -> None:
        super().load_state_dict(state_dict)
        first_group = self.param_groups[0]
        self.ns_algorithm = cast(
            NewtonSchulzAlgorithm,
            first_group.get("ns_algorithm", self.ns_algorithm),
        )
        self.ns_epsilon = cast(
            float,
            first_group.get("ns_epsilon", self.ns_epsilon),
        )
        self.ns_use_kernels = cast(
            bool,
            first_group.get("ns_use_kernels", self.ns_use_kernels),
        )
        self.ns_backend = cast(
            NewtonSchulzBackend,
            first_group.get("ns_backend", self.ns_backend),
        )
        loaded_coefficients = cast(
            Coefficients,
            first_group.get("ns_coefficients", self.ns_coefficients),
        )
        self.ns_coefficients = tuple(
            tuple(float(coefficient) for coefficient in iteration_coefficients)
            for iteration_coefficients in loaded_coefficients
        )
        loaded_reset_iterations = cast(
            Sequence[int],
            first_group.get(
                "gram_newton_schulz_reset_iterations",
                self.gram_newton_schulz_reset_iterations,
            ),
        )
        self.gram_newton_schulz_reset_iterations = tuple(loaded_reset_iterations)
        configuration: ParameterGroup = {
            "ns_algorithm": self.ns_algorithm,
            "ns_epsilon": self.ns_epsilon,
            "ns_use_kernels": self.ns_use_kernels,
            "ns_backend": self.ns_backend,
            "ns_coefficients": self.ns_coefficients,
            "gram_newton_schulz_reset_iterations": (
                self.gram_newton_schulz_reset_iterations
            ),
        }
        self.defaults.update(configuration)
        for parameter_group in self.param_groups:
            for name, value in configuration.items():
                loaded_value = parameter_group.get(name, value)
                if loaded_value != value:
                    raise ValueError(
                        f"Checkpoint has inconsistent Newton--Schulz setting {name}"
                    )
                parameter_group[name] = value
            self.initialize_parameter_group_state(parameter_group)
        self.validate_optimizer_devices()
        self.configure_newton_schulz()

    def add_param_group(self, param_group: ParameterGroupInput) -> None:
        mutable_group = dict(param_group)
        original_group_count = len(self.param_groups)
        original_state_parameters = set(self.state)
        try:
            super().add_param_group(mutable_group)
            if self.state_prepopulation_enabled:
                added_group = self.param_groups[-1]
                self.validate_parameter_group(added_group)
                self.validate_optimizer_devices()
                self.initialize_parameter_group_state(added_group)
        except (TypeError, ValueError, RuntimeError):
            if len(self.param_groups) > original_group_count:
                self.param_groups.pop()
            added_state_parameters = set(self.state) - original_state_parameters
            for parameter in added_state_parameters:
                del self.state[parameter]
            raise

    # ty: ignore[invalid-method-override]
    def step(self, closure: LossClosure | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for parameter_group in self.param_groups:
                algorithm = cast(OptimizerAlgorithm, parameter_group["algorithm"])
                if algorithm == "muon":
                    self.step_muon_group(parameter_group)
                elif algorithm == "adamw":
                    self.step_adamw_group(parameter_group)
                elif algorithm == "lion":
                    self.step_lion_group(parameter_group)
                else:
                    raise ValueError(f"Unknown optimizer algorithm: {algorithm}")
        return loss

    def validate_constructor_values(
        self,
        lr: float,
        momentum: float,
        weight_decay: float,
        betas: tuple[float, float],
        epsilon: float,
        ns_epsilon: float,
        ns_algorithm: NewtonSchulzAlgorithm,
        ns_backend: NewtonSchulzBackend,
        adjust_lr: LearningRateAdjustment,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate must be non-negative, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if epsilon <= 0.0:
            raise ValueError(f"AdamW epsilon must be positive, got {epsilon}")
        if ns_epsilon <= 0.0:
            raise ValueError(
                f"Newton--Schulz epsilon must be positive, got {ns_epsilon}"
            )
        if ns_algorithm not in {
            "auto",
            "gram_newton_schulz",
            "standard_newton_schulz",
        }:
            raise ValueError(f"Unknown Newton--Schulz algorithm: {ns_algorithm}")
        if ns_backend not in {"auto", "cutlass", "triton"}:
            raise ValueError(f"Unknown Newton--Schulz kernel backend: {ns_backend}")
        if adjust_lr is not None and adjust_lr not in {
            "spectral_norm",
            "rms_norm",
        }:
            raise TypeError("adjust_lr must be spectral_norm, rms_norm, or None")

    def accepts_dion3_parameter_groups(self) -> bool:
        return False

    def validate_parameter_group(self, parameter_group: ParameterGroup) -> None:
        algorithm = parameter_group["algorithm"]
        if algorithm not in {"muon", "dion3", "adamw", "lion"}:
            raise ValueError(f"Unknown optimizer algorithm: {algorithm}")
        if algorithm == "dion3" and not self.accepts_dion3_parameter_groups():
            raise ValueError("Dion3 parameter groups require the Dion3 optimizer")
        newton_schulz_configuration: ParameterGroup = {
            "ns_algorithm": self.ns_algorithm,
            "ns_epsilon": self.ns_epsilon,
            "ns_use_kernels": self.ns_use_kernels,
            "ns_backend": self.ns_backend,
            "ns_coefficients": self.ns_coefficients,
            "gram_newton_schulz_reset_iterations": (
                self.gram_newton_schulz_reset_iterations
            ),
        }
        for name, expected_value in newton_schulz_configuration.items():
            if parameter_group[name] != expected_value:
                raise ValueError(
                    f"Newton--Schulz setting {name} must be optimizer-wide"
                )

        parameters = cast(list[Tensor], parameter_group["params"])
        if algorithm in {"muon", "dion3"}:
            invalid_shapes = [
                tuple(parameter.shape) for parameter in parameters if parameter.ndim < 2
            ]
            if invalid_shapes:
                raise ValueError(
                    "Orthogonal optimizer groups require matrix parameters. Put scalar "
                    "and vector parameters in an adamw or lion group. Invalid "
                    f"shapes: {invalid_shapes}"
                )

        self.validate_nonnegative_group_scalar(parameter_group, "lr")
        self.validate_nonnegative_group_scalar(parameter_group, "weight_decay")
        momentum = parameter_group["momentum"]
        if not isinstance(momentum, float) or not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum must be a float in [0, 1), got {momentum}")
        betas = parameter_group["betas"]
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"Betas must be a pair, got {betas}")
        beta_one = betas[0]
        beta_two = betas[1]
        if (
            not isinstance(beta_one, float)
            or not isinstance(beta_two, float)
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
        ):
            raise ValueError(f"Betas must be floats in [0, 1), got {betas}")
        epsilon = parameter_group["epsilon"]
        if not isinstance(epsilon, float) or epsilon <= 0.0:
            raise ValueError(f"Epsilon must be a positive float, got {epsilon}")
        flatten = parameter_group["flatten"]
        if not isinstance(flatten, bool):
            raise TypeError(f"flatten must be bool, got {type(flatten).__name__}")
        adjust_lr = parameter_group["adjust_lr"]
        if adjust_lr is not None and adjust_lr not in {
            "spectral_norm",
            "rms_norm",
        }:
            raise TypeError("adjust_lr must be spectral_norm, rms_norm, or None")

    def validate_nonnegative_group_scalar(
        self,
        parameter_group: ParameterGroup,
        name: str,
    ) -> None:
        value = parameter_group[name]
        if isinstance(value, Tensor):
            if value.numel() != 1 or bool((value < 0).item()):
                raise ValueError(f"{name} must be a non-negative scalar, got {value}")
            return
        if not isinstance(value, float) or value < 0.0:
            raise ValueError(f"{name} must be a non-negative float, got {value}")

    def validate_optimizer_devices(self) -> None:
        devices = {
            local_tensor(parameter).device
            for parameter_group in self.param_groups
            for parameter in cast(list[Tensor], parameter_group["params"])
        }
        if len(devices) > 1:
            raise ValueError("One Muon optimizer cannot span multiple local devices")

    def has_cuda_parameters(self) -> bool:
        cuda_parameters = [
            local_tensor(parameter)
            for parameter_group in self.param_groups
            for parameter in cast(list[Tensor], parameter_group["params"])
            if local_tensor(parameter).is_cuda
        ]
        if not cuda_parameters:
            return False
        compute_capability = torch.cuda.get_device_capability(cuda_parameters[0].device)
        if compute_capability[0] < 8:
            raise RuntimeError("Ampere Muon requires CUDA compute capability 8.0+")
        return True

    def build_newton_schulz(
        self,
        ns_algorithm: NewtonSchulzAlgorithm,
        ns_epsilon: float,
        ns_use_kernels: bool,
        ns_backend: NewtonSchulzBackend,
        ns_coefficients: Coefficients,
        gram_newton_schulz_reset_iterations: Sequence[int],
    ) -> NewtonSchulz:
        if ns_algorithm == "gram_newton_schulz":
            return GramNewtonSchulz(
                ns_epsilon=ns_epsilon,
                ns_use_kernels=ns_use_kernels,
                ns_backend=ns_backend,
                ns_coefficients=ns_coefficients,
                gram_newton_schulz_reset_iterations=(
                    gram_newton_schulz_reset_iterations
                ),
            )
        if ns_algorithm in {"auto", "standard_newton_schulz"}:
            return StandardNewtonSchulz(
                ns_epsilon=ns_epsilon,
                ns_use_kernels=ns_use_kernels,
                ns_backend=ns_backend,
                ns_coefficients=ns_coefficients,
            )
        raise ValueError(f"Unknown Newton--Schulz algorithm: {ns_algorithm}")

    def initialize_parameter_group_state(
        self,
        parameter_group: ParameterGroup,
    ) -> None:
        algorithm = cast(OptimizerAlgorithm, parameter_group["algorithm"])
        parameters = cast(list[Tensor], parameter_group["params"])
        for parameter in parameters:
            parameter_state = self.state[parameter]
            if (
                algorithm in {"muon", "dion3", "lion"}
                and "momentum_buffer" not in parameter_state
            ):
                parameter_state["momentum_buffer"] = torch.zeros_like(parameter)
            if algorithm == "dion3":
                if "variance_neuron" not in parameter_state:
                    parameter_state["variance_neuron"] = torch.zeros_like(
                        parameter[..., :1],
                        dtype=torch.float32,
                    )
                else:
                    variance_neuron = cast(Tensor, parameter_state["variance_neuron"])
                    if variance_neuron.dtype != torch.float32:
                        parameter_state["variance_neuron"] = variance_neuron.float()
            elif algorithm == "adamw":
                if "step" not in parameter_state:
                    parameter_state["step"] = torch.zeros(
                        (),
                        dtype=torch.float32,
                        device=local_tensor(parameter).device,
                    )
                if "exp_avg" not in parameter_state:
                    parameter_state["exp_avg"] = torch.zeros_like(parameter)
                if "exp_avg_sq" not in parameter_state:
                    parameter_state["exp_avg_sq"] = torch.zeros_like(parameter)

    def step_muon_group(self, parameter_group: ParameterGroup) -> None:
        all_parameters = cast(list[Tensor], parameter_group["params"])
        flatten = cast(bool, parameter_group["flatten"])
        validate_gradient_participation(
            all_parameters,
            self.distributed_mesh,
            flatten,
        )
        parameters = [
            parameter for parameter in all_parameters if parameter.grad is not None
        ]
        if not parameters:
            return

        momentum = cast(float, parameter_group["momentum"])
        nesterov = cast(bool, parameter_group["nesterov"])
        matrix_updates: list[Tensor] = []
        global_matrix_shapes: list[tuple[int, ...]] = []

        for parameter in parameters:
            local_gradient = resolve_gradient(parameter)
            self.reject_sparse_or_complex_gradient(local_gradient)
            parameter_state = self.state[parameter]
            momentum_buffer = cast(Tensor, parameter_state["momentum_buffer"])
            local_momentum = local_tensor(momentum_buffer)
            local_momentum.mul_(momentum).add_(
                local_gradient.to(dtype=local_momentum.dtype)
            )
            if nesterov:
                local_update = local_gradient.add(local_momentum, alpha=momentum)
            else:
                local_update = local_momentum
            global_shape = tuple(parameter.shape)
            matrix_updates.append(
                self.reshape_for_orthogonalization(
                    local_update,
                    global_shape,
                    flatten,
                )
            )
            global_matrix_shapes.append(self.matrix_shape(global_shape, flatten))

        orthogonalized_updates = orthogonalize_parameter_updates(
            parameters,
            matrix_updates,
            global_matrix_shapes,
            self.newton_schulz,
            self.distributed_mesh,
            flatten,
        )

        learning_rate = cast(LearningRate, parameter_group["lr"])
        weight_decay = cast(LearningRate, parameter_group["weight_decay"])
        adjust_lr = cast(LearningRateAdjustment, parameter_group["adjust_lr"])
        for parameter_index, parameter in enumerate(parameters):
            adjusted_learning_rate = self.adjusted_learning_rate(
                learning_rate,
                global_matrix_shapes[parameter_index],
                adjust_lr,
            )
            local_parameter = local_tensor(parameter)
            local_update = orthogonalized_updates[parameter_index].reshape(
                local_parameter.shape
            )
            if not isinstance(weight_decay, float) or weight_decay != 0.0:
                local_parameter.mul_(1 - learning_rate * weight_decay)
            local_parameter.sub_(local_update * adjusted_learning_rate)

    def reshape_for_orthogonalization(
        self,
        update: Tensor,
        global_shape: tuple[int, ...],
        flatten: bool,
    ) -> Tensor:
        if not flatten or len(global_shape) == 2:
            return update
        return rearrange(update, "output ... -> output (...)")

    def matrix_shape(
        self,
        global_shape: tuple[int, ...],
        flatten: bool,
    ) -> tuple[int, ...]:
        if not flatten or len(global_shape) == 2:
            return global_shape
        input_dimension = 1
        for dimension_size in global_shape[1:]:
            input_dimension *= dimension_size
        return (global_shape[0], input_dimension)

    def adjusted_learning_rate(
        self,
        learning_rate: LearningRate,
        matrix_shape: tuple[int, ...],
        adjustment: LearningRateAdjustment,
    ) -> LearningRate:
        if adjustment is None:
            return learning_rate
        rows = matrix_shape[-2]
        columns = matrix_shape[-1]
        if adjustment == "spectral_norm":
            return learning_rate * sqrt(rows / columns)
        if adjustment == "rms_norm":
            return learning_rate * 0.2 * sqrt(max(rows, columns))
        raise ValueError(f"Unknown learning-rate adjustment: {adjustment}")

    def step_adamw_group(self, parameter_group: ParameterGroup) -> None:
        learning_rate = cast(LearningRate, parameter_group["lr"])
        weight_decay = cast(LearningRate, parameter_group["weight_decay"])
        betas = cast(tuple[float, float], parameter_group["betas"])
        epsilon = cast(float, parameter_group["epsilon"])
        parameters = cast(list[Tensor], parameter_group["params"])

        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = resolve_gradient(parameter)
            self.reject_sparse_or_complex_gradient(gradient)
            parameter_state = self.state[parameter]
            step_tensor = cast(Tensor, parameter_state["step"])
            exponential_average = local_tensor(cast(Tensor, parameter_state["exp_avg"]))
            exponential_average_squared = local_tensor(
                cast(Tensor, parameter_state["exp_avg_sq"])
            )
            local_parameter = local_tensor(parameter)

            step_tensor.add_(1)
            step = float(step_tensor.item())
            gradient = gradient.to(dtype=exponential_average.dtype)
            exponential_average.mul_(betas[0]).add_(gradient, alpha=1 - betas[0])
            exponential_average_squared.mul_(betas[1]).addcmul_(
                gradient,
                gradient,
                value=1 - betas[1],
            )
            if not isinstance(weight_decay, float) or weight_decay != 0.0:
                local_parameter.mul_(1 - learning_rate * weight_decay)
            bias_correction_one = 1 - betas[0] ** step
            bias_correction_two = 1 - betas[1] ** step
            denominator = (
                exponential_average_squared.sqrt()
                .div_(sqrt(bias_correction_two))
                .add_(epsilon)
            )
            scalar_update = exponential_average.div(denominator).mul(
                learning_rate / bias_correction_one
            )
            local_parameter.sub_(scalar_update.to(dtype=local_parameter.dtype))

    def step_lion_group(self, parameter_group: ParameterGroup) -> None:
        learning_rate = cast(LearningRate, parameter_group["lr"])
        weight_decay = cast(LearningRate, parameter_group["weight_decay"])
        betas = cast(tuple[float, float], parameter_group["betas"])
        parameters = cast(list[Tensor], parameter_group["params"])

        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = resolve_gradient(parameter)
            self.reject_sparse_or_complex_gradient(gradient)
            parameter_state = self.state[parameter]
            momentum_buffer = local_tensor(
                cast(Tensor, parameter_state["momentum_buffer"])
            )
            local_parameter = local_tensor(parameter)
            gradient = gradient.to(dtype=momentum_buffer.dtype)
            update = momentum_buffer.mul(betas[0]).add(
                gradient,
                alpha=1 - betas[0],
            )
            if not isinstance(weight_decay, float) or weight_decay != 0.0:
                local_parameter.mul_(1 - learning_rate * weight_decay)
            local_parameter.sub_(
                update.sign().mul(learning_rate).to(dtype=local_parameter.dtype)
            )
            momentum_buffer.mul_(betas[1]).add_(
                gradient,
                alpha=1 - betas[1],
            )

    def reject_sparse_or_complex_gradient(self, gradient: Tensor) -> None:
        if gradient.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")
        if gradient.is_complex():
            raise RuntimeError("Muon does not support complex gradients")
