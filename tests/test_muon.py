from collections.abc import Generator
from copy import deepcopy

import pytest
import torch
from torch import Tensor

from gram_newton_schulz_ampere import GramNewtonSchulz, Muon, StandardNewtonSchulz


@pytest.fixture(autouse=True)
def cuda_default_device() -> Generator[None, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("cuda"):
        yield


class IdentityOrthogonalizer:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def __call__(self, matrix: Tensor) -> Tensor:
        self.input_shapes.append(tuple(matrix.shape))
        return matrix


def test_muon_momentum_nesterov_and_native_state_shape() -> None:
    first_gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second_gradient = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    parameter = torch.nn.Parameter(torch.zeros_like(first_gradient))
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=False,
        adjust_lr=None,
    )
    optimizer.newton_schulz = IdentityOrthogonalizer()

    parameter.grad = first_gradient.clone()
    optimizer.step()
    parameter.grad = second_gradient.clone()
    optimizer.step()

    expected_momentum = 0.5 * first_gradient + second_gradient
    expected_parameter = -0.1 * first_gradient - 0.1 * expected_momentum
    torch.testing.assert_close(parameter, expected_parameter)
    momentum = optimizer.state[parameter]["momentum_buffer"]
    torch.testing.assert_close(momentum, expected_momentum)
    assert momentum.shape == parameter.shape

    nesterov_parameter = torch.nn.Parameter(torch.zeros_like(first_gradient))
    nesterov_optimizer = Muon(
        [nesterov_parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=True,
        adjust_lr=None,
    )
    nesterov_optimizer.newton_schulz = IdentityOrthogonalizer()
    nesterov_parameter.grad = first_gradient.clone()
    nesterov_optimizer.step()
    torch.testing.assert_close(
        nesterov_parameter,
        -0.1 * (first_gradient + 0.5 * first_gradient),
    )


def test_conv_flatten_uses_einops_and_preserves_state_shape() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 2, 2, 2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = Muon(
        [{"params": [parameter], "flatten": True}],
        momentum=0.0,
        nesterov=False,
        adjust_lr=None,
    )
    orthogonalizer = IdentityOrthogonalizer()
    optimizer.newton_schulz = orthogonalizer
    optimizer.step()

    assert orthogonalizer.input_shapes == [(1, 3, 8)]
    assert optimizer.state[parameter]["momentum_buffer"].shape == parameter.shape


def test_closure_runs_once_with_grad_and_returns_loss() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = Muon(
        [parameter],
        momentum=0.0,
        nesterov=False,
    )
    optimizer.newton_schulz = IdentityOrthogonalizer()
    grad_modes: list[bool] = []

    def closure() -> float:
        grad_modes.append(torch.is_grad_enabled())
        return float(parameter.square().sum().detach())

    loss = optimizer.step(closure)
    assert loss is not None
    assert grad_modes == [True]


def test_adamw_group_matches_torch_and_prepopulates_state() -> None:
    initial = torch.tensor([1.0, -2.0, 3.0])
    ours = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.clone())
    ours_optimizer = Muon(
        [{"params": [ours], "algorithm": "adamw"}],
        lr=0.03,
        weight_decay=0.1,
        betas=(0.8, 0.9),
        epsilon=1e-6,
    )
    reference_optimizer = torch.optim.AdamW(
        [reference],
        lr=0.03,
        weight_decay=0.1,
        betas=(0.8, 0.9),
        eps=1e-6,
    )
    assert set(ours_optimizer.state[ours]) == {"step", "exp_avg", "exp_avg_sq"}

    for gradient in (
        torch.tensor([0.5, -0.25, 1.0]),
        torch.tensor([-0.5, 0.75, 0.25]),
    ):
        ours.grad = gradient.clone()
        reference.grad = gradient.clone()
        ours_optimizer.step()
        reference_optimizer.step()
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-7)


def test_decay_uses_base_lr_and_update_uses_adjusted_lr() -> None:
    parameter = torch.nn.Parameter(torch.ones(8, 2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.2,
        adjust_lr="spectral_norm",
    )
    optimizer.newton_schulz = IdentityOrthogonalizer()
    optimizer.step()

    expected = torch.ones_like(parameter) * (1 - 0.1 * 0.2)
    expected.sub_(torch.ones_like(parameter) * 0.1 * 2.0)
    torch.testing.assert_close(parameter, expected)


def test_actual_gram_update_matches_direct_call() -> None:
    generator = torch.Generator(device="cuda").manual_seed(67)
    gradient = torch.randn(8, 4, generator=generator)
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    parameter.grad = gradient.clone()
    optimizer = Muon(
        [parameter],
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        adjust_lr=None,
    )
    assert isinstance(optimizer.newton_schulz, GramNewtonSchulz)
    optimizer.step()

    expected = -0.05 * GramNewtonSchulz()(gradient)
    torch.testing.assert_close(parameter, expected)


def test_muon_rejects_vector_sparse_and_complex_gradients() -> None:
    with pytest.raises(ValueError, match="matrix parameters"):
        Muon([torch.nn.Parameter(torch.ones(3))])

    sparse_parameter = torch.nn.Parameter(torch.zeros(3, 3))
    sparse_parameter.grad = torch.eye(3).to_sparse()
    sparse_optimizer = Muon([sparse_parameter])
    with pytest.raises(RuntimeError, match="sparse"):
        sparse_optimizer.step()

    complex_parameter = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.complex64))
    complex_parameter.grad = torch.ones_like(complex_parameter)
    complex_optimizer = Muon([complex_parameter])
    with pytest.raises(RuntimeError, match="complex"):
        complex_optimizer.step()


def test_state_dict_resume_scheduler_and_add_param_group() -> None:
    generator = torch.Generator(device="cuda").manual_seed(71)
    initial = torch.randn(4, 3, generator=generator)
    first_gradient = torch.randn(4, 3, generator=generator)
    second_gradient = torch.randn(4, 3, generator=generator)
    third_gradient = torch.randn(4, 3, generator=generator)
    uninterrupted = torch.nn.Parameter(initial.clone())
    checkpointed = torch.nn.Parameter(initial.clone())
    uninterrupted_optimizer = Muon(
        [uninterrupted],
        lr=0.1,
        momentum=0.5,
        nesterov=True,
        adjust_lr=None,
    )
    checkpointed_optimizer = Muon(
        [checkpointed],
        lr=0.1,
        momentum=0.5,
        nesterov=True,
        adjust_lr=None,
    )
    uninterrupted_optimizer.newton_schulz = IdentityOrthogonalizer()
    checkpointed_optimizer.newton_schulz = IdentityOrthogonalizer()
    uninterrupted.grad = first_gradient.clone()
    checkpointed.grad = first_gradient.clone()
    uninterrupted_optimizer.step()
    checkpointed_optimizer.step()

    resumed = torch.nn.Parameter(checkpointed.detach().clone())
    resumed_optimizer = Muon(
        [resumed],
        lr=0.1,
        momentum=0.5,
        nesterov=True,
        adjust_lr=None,
    )
    resumed_optimizer.load_state_dict(deepcopy(checkpointed_optimizer.state_dict()))
    resumed_optimizer.newton_schulz = IdentityOrthogonalizer()
    scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer,
        step_size=1,
        gamma=0.5,
    )
    uninterrupted.grad = second_gradient.clone()
    resumed.grad = second_gradient.clone()
    uninterrupted_optimizer.step()
    resumed_optimizer.step()
    scheduler.step()
    uninterrupted_optimizer.param_groups[0]["lr"] = 0.05
    uninterrupted.grad = third_gradient.clone()
    resumed.grad = third_gradient.clone()
    uninterrupted_optimizer.step()
    resumed_optimizer.step()
    torch.testing.assert_close(resumed, uninterrupted)

    vector = torch.nn.Parameter(torch.ones(3))
    resumed_optimizer.add_param_group({"params": [vector], "algorithm": "adamw"})
    assert set(resumed_optimizer.state[vector]) == {
        "step",
        "exp_avg",
        "exp_avg_sq",
    }


def test_compiled_gram_matches_eager_tall_matrix() -> None:
    generator = torch.Generator(device="cuda").manual_seed(131)
    matrix = torch.randn(
        (2, 256, 64),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = GramNewtonSchulz(ns_compile=False)(matrix)
    compiled = GramNewtonSchulz(ns_compile=True)
    actual = compiled(matrix)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, compiled(matrix))


def test_newton_schulz_configuration_resume_and_square_fallback() -> None:
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = Muon(
        [parameter],
        ns_algorithm="standard_newton_schulz",
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    checkpoint = deepcopy(optimizer.state_dict())

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = Muon(
        [resumed_parameter],
        ns_algorithm="gram_newton_schulz",
    )
    resumed.load_state_dict(checkpoint)
    assert resumed.ns_algorithm == "standard_newton_schulz"
    assert isinstance(resumed.newton_schulz, StandardNewtonSchulz)

    missing_state_checkpoint = deepcopy(checkpoint)
    missing_state_checkpoint["state"] = {}
    resumed.load_state_dict(missing_state_checkpoint)
    assert "momentum_buffer" in resumed.state[resumed_parameter]

    square = torch.randn(4, 4)
    gram = GramNewtonSchulz()(square)
    standard = StandardNewtonSchulz()(square)
    torch.testing.assert_close(gram, standard, rtol=0, atol=0)


def test_add_param_group_is_atomic_and_callable_adjustment_is_rejected() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = Muon([parameter])
    original_group_count = len(optimizer.param_groups)
    original_state_parameters = set(optimizer.state)
    invalid_parameter = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="matrix"):
        optimizer.add_param_group({"params": [invalid_parameter], "algorithm": "muon"})
    assert len(optimizer.param_groups) == original_group_count
    assert set(optimizer.state) == original_state_parameters

    with pytest.raises(TypeError, match="adjust_lr"):
        Muon(
            [torch.nn.Parameter(torch.zeros(2, 2))],
            adjust_lr=lambda learning_rate, matrix_shape: learning_rate,
        )
