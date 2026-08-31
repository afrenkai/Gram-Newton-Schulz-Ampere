from collections.abc import Callable
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from einops import rearrange
from coefficients import POLAR_EXPRESS_COEFFICIENTS, YOU_COEFFICIENTS
from newton_schulz import GramNewtonSchulz, StandardNewtonSchulz
from muon_utils.muon_matrix_split_utils import (
    get_newton_schulz_inputs_from_gradients,
    reconstruct_update_from_newton_schulz_outputs,
    scale_newton_schulz_outputs_with_adjusted_lr,
)
from muon_utils.muon_opt_utils import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    create_param_batches,
    get_or_initialize_muon_state,
    muon_update_post_orthogonalize,
    muon_update_pre_orthogonalize,
)


class Muon(Optimizer):
    """
    Fast Muon implementation for Gram Newton-Schulz and standard Newton-Schulz.
    Supports:
        - Custom Ampere Triton GEMM kernels for accelerated Newton-Schulz
        - Auxiliary scalar optimizer for non-Muon weight updates, supporting LR scheduling.
        - Custom NS coefficients, with default POLAR_EXPRESS_COEFFICIENTS from newton-schulz/coefficients.py
        - Custom weight splitting logic via lambda functions during preprocessing before Newton-Schulz
        - Custom Muon LR adjustment function
        - Single GPU training only

    Args:
        params: Parameter groups. Each group can specify:
            - param_split_fn: Function to split a parameter into submatrices before orthogonalization (e.g., split Wqkv into Wq, Wk, Wv)
            - param_recombine_fn: Function to recombine submatrices after orthogonalization back into original parameter shape
            - 3D weights are by default treated as batched 2D weights, with the first dimension being the batch dimension
            - See example.py for example usage
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay coefficient (default: 0.1)
        momentum: Momentum factor (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        adjust_lr: Learning rate adjustment method. Options:
            - "rms_norm": Scale by sqrt(max(fan_out, fan_in)) for constant element-wise RMS norm
            - "spectral_norm": Scale from spectral norm 1 to RMS operator norm 1
            - Callable: Custom function taking (lr, param_shape) -> adjusted_lr
            - None: No adjustment
            (default: "rms_norm")
        ns_coefficients: List of 3-coefficient tuples for each Newton-Schulz iteration.
            Each tuple contains [a, b, c] coefficients for the iteration formula.
            If ns_coefficients_preset is provided, this parameter is ignored.
            (default: POLAR_EXPRESS_COEFFICIENTS from newton-schulz/coefficients.py)
        ns_coefficients_preset: Select a coefficient preset by name. Options:
            - "POLAR_EXPRESS_COEFFICIENTS": Polar Express coefficients with /1.01 scaling
            - "YOU_COEFFICIENTS": YOU coefficients
            - None: Use ns_coefficients parameter or default
            (default: None)
        ns_algorithm: Newton-Schulz algorithm variant. Options:
            - "gram_newton_schulz": Gram Newton-Schulz iteration with optional resets
            - "standard_newton_schulz": Standard Newton-Schulz iteration
            (default: "gram_newton_schulz")
        ns_max_batch_size: Maximum number of matrices per Newton-Schulz call.
            When a shape group has more matrices than this, they are processed in
            micro-batches to reduce peak GPU memory. None means no limit (all
            matrices of the same shape are processed in one call).
            (default: None)
        ns_epsilon: Epsilon for Frobenius normalization before orthogonalization (default: 1e-7)
        ns_use_kernels: Use custom CUDA kernels if available (requires compute capability 8.0+) (default: True)
        gram_newton_schulz_num_restarts: Number of restarts for Gram Newton-Schulz. Restart positions are automatically tuned during initialization. Ignored if gram_newton_schulz_restart_iterations is provided. (default: 1)
        gram_newton_schulz_restart_iterations: Manual restart positions for Gram Newton-Schulz. If "2" is an entry, the user wants a restart after the 2nd iteration. If provided, auto-tuning is skipped. (default: None)
        scalar_optimizer: Optional secondary optimizer for non-matrix parameters (default: None)

    Example: examples/example.py
    """

    def __init__(
        self,
        # Muon
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        adjust_lr: str | Callable[[float, tuple[int, ...]], float] | None = "rms_norm",
        ns_coefficients: list[tuple[float, float, float] | list[float]] | None = None,
        ns_coefficients_preset: str | None = None,
        ns_algorithm: str = "gram_newton_schulz",
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_max_batch_size: int | None = None,
        gram_newton_schulz_num_restarts: int = 1,
        gram_newton_schulz_restart_iterations: (
            list[int] | tuple[int, ...] | None
        ) = None,
        scalar_optimizer: Optimizer | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Learning rate must be positive: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay can't be negative: {weight_decay}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Momentum must be in [0, 1): {momentum}")
        if ns_epsilon <= 0.0:
            raise ValueError(
                f"Newton-Schulz epsilon for normalization must be positive: {ns_epsilon}"
            )

        if ns_coefficients_preset is not None:
            preset_map = {
                "POLAR_EXPRESS_COEFFICIENTS": POLAR_EXPRESS_COEFFICIENTS,
                "YOU_COEFFICIENTS": YOU_COEFFICIENTS,
            }
            if ns_coefficients_preset not in preset_map:
                raise ValueError(
                    f"Invalid ns_coefficients_preset: {ns_coefficients_preset}. Must be one of: {list(preset_map.keys())}"
                )
            ns_coefficients = preset_map[ns_coefficients_preset]
        elif ns_coefficients is None:
            ns_coefficients = POLAR_EXPRESS_COEFFICIENTS

        if ns_algorithm not in ("gram_newton_schulz", "standard_newton_schulz"):
            raise ValueError(
                f"Invalid ns_algorithm: {ns_algorithm}. Must be 'gram_newton_schulz' or 'standard_newton_schulz'."
            )

        if (
            not isinstance(gram_newton_schulz_num_restarts, int)
            or gram_newton_schulz_num_restarts < 0
        ):
            raise ValueError(
                f"gram_newton_schulz_num_restarts must be a non-negative integer, got {gram_newton_schulz_num_restarts}"
            )

        ns_coefficients = [
            (
                list(coef)
                if hasattr(coef, "__iter__") and not isinstance(coef, str)
                else coef
            )
            for coef in ns_coefficients
        ]

        for i, coef in enumerate(ns_coefficients):
            if len(coef) != 3:
                raise ValueError(
                    f"Each iteration must have exactly 3 Newton-Schulz coefficients, got {len(coef)} at iteration {i}"
                )

        if ns_max_batch_size is not None and (
            not isinstance(ns_max_batch_size, int) or ns_max_batch_size < 1
        ):
            raise ValueError(
                f"ns_max_batch_size must be a positive integer or None, got {ns_max_batch_size}"
            )

        self.ns_coefficients = ns_coefficients
        self.ns_algorithm = ns_algorithm
        self.ns_epsilon = ns_epsilon
        self.ns_max_batch_size = ns_max_batch_size

        if gram_newton_schulz_restart_iterations is not None:
            self.gram_newton_schulz_reset_iterations = list(
                gram_newton_schulz_restart_iterations
            )
        elif ns_algorithm == "gram_newton_schulz":
            coefficient_count = len(ns_coefficients)
            if gram_newton_schulz_num_restarts >= coefficient_count:
                raise ValueError(
                    "gram_newton_schulz_num_restarts must be less than the "
                    "coefficient count"
                )
            self.gram_newton_schulz_reset_iterations = [
                (restart_index + 1)
                * coefficient_count
                // (gram_newton_schulz_num_restarts + 1)
                for restart_index in range(gram_newton_schulz_num_restarts)
            ]
        else:
            self.gram_newton_schulz_reset_iterations = []

        self.ns_use_kernels = False
        if ns_use_kernels and torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            compute_capability = capability[0] * 10 + capability[1]
            self.ns_use_kernels = compute_capability >= 80

        self.scalar_optimizer = scalar_optimizer
        self._muon_param_groups = None
        self._combined_param_groups = None
        if self.scalar_optimizer is not None:

            @torch.compile(fullgraph=False)
            def compiled_scalar_step():
                self.scalar_optimizer.step()

            self.compiled_scalar_step = compiled_scalar_step
        else:
            self.compiled_scalar_step = None

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "adjust_lr": adjust_lr,
        }
        super().__init__(params, defaults)
        if self.scalar_optimizer is not None:
            self._combined_param_groups = (
                self._muon_param_groups + self.scalar_optimizer.param_groups
            )
        else:
            self._combined_param_groups = self._muon_param_groups

        if self.ns_algorithm == "gram_newton_schulz":
            self.newton_schulz = GramNewtonSchulz(
                ns_epsilon=self.ns_epsilon,
                ns_use_kernels=self.ns_use_kernels,
                ns_coefficients=self.ns_coefficients,
                gram_newton_schulz_reset_iterations=self.gram_newton_schulz_reset_iterations,
            )
        elif self.ns_algorithm == "standard_newton_schulz":
            self.newton_schulz = StandardNewtonSchulz(
                ns_epsilon=self.ns_epsilon,
                ns_use_kernels=self.ns_use_kernels,
                ns_coefficients=self.ns_coefficients,
            )
        else:
            raise ValueError(
                f"Invalid ns_algorithm: {self.ns_algorithm}. Must be 'gram_newton_schulz' or 'standard_newton_schulz'."
            )

    @property
    def param_groups(self):
        if self._combined_param_groups is None:
            return (
                self._muon_param_groups if self._muon_param_groups is not None else []
            )
        return self._combined_param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._muon_param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.muon_step(self._muon_param_groups)

        if self.compiled_scalar_step is not None:
            self.compiled_scalar_step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)

        if self.scalar_optimizer is not None:
            self.scalar_optimizer.zero_grad(set_to_none=set_to_none)

    def muon_step(self, param_groups: list[dict]):
        for group in param_groups:
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports 2D matrix parameters or 3D parameters that will be treated as a batch of 2D matrices."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            adjust_lr = group["adjust_lr"]
            if isinstance(adjust_lr, str):
                if adjust_lr == "rms_norm":
                    adjust_lr_fn = adjust_lr_rms_norm
                elif adjust_lr == "spectral_norm":
                    adjust_lr_fn = adjust_lr_spectral_norm
                else:
                    raise ValueError(
                        f"Invalid adjust_lr: {adjust_lr}. Must be 'rms_norm', 'spectral_norm', or a callable."
                    )
            elif callable(adjust_lr) or adjust_lr is None:
                adjust_lr_fn = adjust_lr
            else:
                raise TypeError(f"Invalid adjust_lr type: {type(adjust_lr)}")

            muon_batch_update_args = {
                "lr": group["lr"],
                "momentum": group["momentum"],
                "weight_decay": group["weight_decay"],
                "nesterov": group["nesterov"],
                "adjust_lr_fn": adjust_lr_fn,
                "param_split_fn": group.get("param_split_fn"),
                "param_recombine_fn": group.get("param_recombine_fn"),
            }

            for params in create_param_batches(group_params):
                gradients = [p.grad for p in params]
                states = [get_or_initialize_muon_state(self.state, p) for p in params]
                momentums = [s["momentum"] for s in states]
                self.muon_batch_update(
                    params, gradients, momentums, **muon_batch_update_args
                )

    def muon_batch_update(
        self,
        params: list[Tensor],
        gradients: list[Tensor],
        momentums: list[Tensor],
        lr: Tensor,
        momentum: Tensor,
        weight_decay: Tensor,
        nesterov: bool,
        adjust_lr_fn: Callable | None,
        param_split_fn: Callable | None,
        param_recombine_fn: Callable | None,
    ):
        assert (
            len(params) == len(gradients) == len(momentums)
        ), "Number of parameters, gradients, and momentums for Muon must match"

        if (param_split_fn is None) != (param_recombine_fn is None):
            raise ValueError(
                "param_split_fn and param_recombine_fn must both be provided or both be None"
            )

        ns_inputs = muon_update_pre_orthogonalize(
            G=gradients,
            M=momentums,
            momentum=momentum,
            nesterov=nesterov,
        )

        if len(ns_inputs) > 0:
            ns_inputs_by_shape, shape_indices, split_metadata = (
                get_newton_schulz_inputs_from_gradients(ns_inputs, param_split_fn)
            )

            orthogonalized_by_shape = {}
            max_bs = self.ns_max_batch_size
            for shape, ns_inputs_for_shape in ns_inputs_by_shape.items():
                if max_bs is None or len(ns_inputs_for_shape) <= max_bs:
                    batched_input = torch.stack(ns_inputs_for_shape, dim=0)
                    orthogonalized_batched = self.newton_schulz(batched_input)
                    orthogonalized_by_shape[shape] = orthogonalized_batched.clone()
                else:
                    total = len(ns_inputs_for_shape)
                    first_end = min(max_bs, total)
                    first_chunk = torch.stack(ns_inputs_for_shape[:first_end], dim=0)
                    first_out = self.newton_schulz(first_chunk).clone()
                    full_output = first_out.new_empty((total, *first_out.shape[1:]))
                    full_output[:first_end].copy_(first_out)
                    for i in range(first_end, total, max_bs):
                        chunk = torch.stack(ns_inputs_for_shape[i : i + max_bs], dim=0)
                        chunk_out = self.newton_schulz(chunk).clone()
                        full_output[i : i + chunk_out.shape[0]].copy_(chunk_out)
                    orthogonalized_by_shape[shape] = full_output

            orthogonalized_by_shape = scale_newton_schulz_outputs_with_adjusted_lr(
                orthogonalized_by_shape, lr, adjust_lr_fn
            )
            orthogonalized = reconstruct_update_from_newton_schulz_outputs(
                orthogonalized_by_shape,
                shape_indices,
                split_metadata,
                param_recombine_fn,
            )
        else:
            orthogonalized = []

        muon_update_post_orthogonalize(params, orthogonalized, lr, weight_decay)


import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.optim import Optimizer

from ampere_ns_interface import TORCH_BE, TRITON_BE



class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        *,
        use_triton: bool | None = None,
    ) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        if use_triton is None:
            use_triton = TRITON_BE is not None and any(
                parameter.is_cuda for group in self.param_groups for parameter in group["params"]
            )
        self.newton_schulz = NewtonSchulz(use_triton=use_triton)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, wd = group["nesterov"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                orig_shape = g.shape
                if g.ndim == 1:
                    p.data.mul_(1 - lr * wd).add_(g, alpha=-lr)
                    continue
                if g.ndim == 4:
                    g = rearrange(g, "out_ch in_ch kh kw -> out_ch (in_ch kh kw)")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf

                g = self.newton_schulz(g)
                if wd:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(g.reshape(orig_shape), alpha=-lr)
