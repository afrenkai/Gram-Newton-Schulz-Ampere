import copy
from collections.abc import Generator

import pytest
import torch
from torch import Tensor

from gram_newton_schulz_ampere import Dion3
from gram_newton_schulz_ampere.dion3_update import (
    apply_dion3_update,
    normalize_dion3_rows,
    select_dion3_rows,
)
from gram_newton_schulz_ampere.kernels.dion3_cuda import (
    normalize_apply_rows_cuda,
    normalize_apply_rows_cuda_batch,
    select_dion3_rows_cuda,
    select_dion3_rows_cuda_batch,
)


@pytest.fixture(autouse=True)
def cuda_default_device() -> Generator[None, None, None]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("cuda"):
        yield


class IdentityOrthogonalizer:
    def __call__(self, matrix: Tensor) -> Tensor:
        return matrix


def test_dion3_update_primitives() -> None:
    momentum = torch.zeros(4, 2)
    gradient = torch.tensor([[1.0, 0.0], [4.0, 4.0], [2.0, 0.0], [0.5, 0.0]])
    selected, indices = select_dion3_rows(
        momentum,
        gradient,
        fraction=0.5,
        error_feedback_decay=0.5,
    )
    assert set(indices.tolist()) == {1, 2}
    assert torch.equal(momentum[0], gradient[0])
    assert torch.equal(momentum[3], gradient[3])
    assert torch.equal(momentum[1], gradient[1] * 0.5)
    assert torch.equal(momentum[2], gradient[2] * 0.5)

    variance = torch.zeros(4, 1)
    normalized = normalize_dion3_rows(
        selected,
        variance,
        indices,
        muon_beta2=0.9,
        epsilon=1e-8,
    )
    expected_rows = {
        1: torch.tensor([3.0, 3.0]),
        2: torch.tensor([4.2426405, 0.0]),
    }
    for selected_index, row_index in enumerate(indices.tolist()):
        assert torch.allclose(
            normalized[selected_index].float(),
            expected_rows[row_index],
            atol=2e-2,
        )
    assert torch.isclose(variance[1], torch.tensor([1.6])).all()
    assert torch.isclose(variance[2], torch.tensor([0.2])).all()

    parameter = torch.ones(4, 2)
    apply_dion3_update(
        parameter,
        normalized,
        indices,
        learning_rate=0.1,
        weight_decay=0.2,
        adjusted_learning_rate=0.1,
    )
    expected = torch.full((4, 2), 0.98)
    for selected_index, row_index in enumerate(indices.tolist()):
        expected[row_index].add_(normalized[selected_index].float(), alpha=-0.1)
    assert torch.allclose(parameter, expected, atol=2e-3)

    batched_momentum = torch.zeros(2, 4, 2)
    batched_gradient = torch.stack((gradient, gradient.flip(0)))
    batched_selected, batched_indices = select_dion3_rows(
        batched_momentum,
        batched_gradient,
        fraction=0.5,
        error_feedback_decay=0.5,
    )
    batched_variance = torch.zeros(2, 4, 1)
    batched_normalized = normalize_dion3_rows(
        batched_selected,
        batched_variance,
        batched_indices,
        muon_beta2=0.9,
        epsilon=1e-8,
    )
    batched_parameter = torch.ones(2, 4, 2)
    apply_dion3_update(
        batched_parameter,
        batched_normalized,
        batched_indices,
        learning_rate=0.1,
        weight_decay=0.0,
        adjusted_learning_rate=0.1,
    )
    assert torch.isfinite(batched_parameter).all()
    assert batched_variance.count_nonzero() == 4

    full_momentum = torch.zeros(2, 4, 2)
    full_selected, full_indices = select_dion3_rows(
        full_momentum,
        batched_gradient,
        fraction=1.0,
        error_feedback_decay=0.5,
    )
    assert torch.equal(full_indices, torch.arange(4).expand(2, 4))
    assert torch.equal(full_momentum, batched_gradient * 0.5)
    torch.testing.assert_close(full_selected.float(), batched_gradient)


def test_dion3_optimizer_state_closure_and_scalar_routing() -> None:
    matrix = torch.nn.Parameter(torch.zeros(4, 2))
    vector = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Dion3(
        [
            {"params": [matrix], "algorithm": "dion3"},
            {"params": [vector], "algorithm": "adamw"},
        ],
        lr=0.1,
        fraction=0.5,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
    )
    optimizer.newton_schulz = IdentityOrthogonalizer()
    closure_calls = 0

    def closure() -> float:
        nonlocal closure_calls
        closure_calls += 1
        matrix.grad = torch.tensor([[1.0, 0.0], [4.0, 4.0], [2.0, 0.0], [0.5, 0.0]])
        vector.grad = torch.tensor([0.25, -0.5])
        loss = matrix.square().sum() + vector.square().sum()
        return float(loss.detach())

    loss = optimizer.step(closure)
    assert closure_calls == 1
    assert loss is not None
    assert torch.allclose(matrix[1], torch.tensor([-0.3, -0.3]), atol=2e-3)
    assert torch.allclose(matrix[2], torch.tensor([-0.425, 0.0]), atol=2e-3)
    assert not torch.equal(vector, torch.tensor([1.0, -2.0]))
    assert set(optimizer.state[matrix]) == {"momentum_buffer", "variance_neuron"}
    assert set(optimizer.state[vector]) == {"step", "exp_avg", "exp_avg_sq"}

    saved_state = copy.deepcopy(optimizer.state_dict())
    resumed_matrix = torch.nn.Parameter(matrix.detach().clone())
    resumed_vector = torch.nn.Parameter(vector.detach().clone())
    resumed = Dion3(
        [
            {"params": [resumed_matrix], "algorithm": "dion3"},
            {"params": [resumed_vector], "algorithm": "adamw"},
        ],
        lr=0.1,
        fraction=0.5,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
    )
    resumed.load_state_dict(saved_state)
    resumed.newton_schulz = IdentityOrthogonalizer()
    for target_matrix, target_vector, target_optimizer in (
        (matrix, vector, optimizer),
        (resumed_matrix, resumed_vector, resumed),
    ):
        target_matrix.grad = torch.ones_like(target_matrix)
        target_vector.grad = torch.tensor([0.1, -0.1])
        target_optimizer.step()
    assert torch.equal(matrix, resumed_matrix)
    assert torch.equal(vector, resumed_vector)


def test_dion3_validation_and_add_param_group() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = Dion3(
        [parameter],
        fraction=0.5,
    )
    added = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer.add_param_group({"params": [added]})
    assert optimizer.param_groups[-1]["algorithm"] == "dion3"
    assert "variance_neuron" in optimizer.state[added]

    batched_matrix = torch.nn.Parameter(torch.zeros(2, 3, 4, 5))
    batched_optimizer = Dion3(
        [batched_matrix],
        fraction=0.5,
        adjust_lr=None,
    )
    batched_optimizer.newton_schulz = IdentityOrthogonalizer()
    batched_matrix.grad = torch.randn_like(batched_matrix)
    batched_optimizer.step()
    assert batched_optimizer.state[batched_matrix]["variance_neuron"].shape == (
        2,
        3,
        4,
        1,
    )
    assert torch.isfinite(batched_matrix).all()

    bfloat16_matrix = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.bfloat16))
    bfloat16_optimizer = Dion3([bfloat16_matrix])
    assert (
        bfloat16_optimizer.state[bfloat16_matrix]["variance_neuron"].dtype
        == torch.float32
    )

    invalid_fraction = torch.nn.Parameter(torch.zeros(2, 2))
    try:
        Dion3([invalid_fraction], fraction=0.0)
    except ValueError as error:
        assert "fraction" in str(error)
    else:
        raise AssertionError("Expected invalid fraction rejection")

    invalid_shape = torch.nn.Parameter(torch.zeros(2))
    try:
        Dion3([invalid_shape])
    except ValueError as error:
        assert "matrix" in str(error)
    else:
        raise AssertionError("Expected invalid shape rejection")


def test_dion3_rejected_groups_are_atomic() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = Dion3([parameter])
    original_group_count = len(optimizer.param_groups)
    original_state_parameters = set(optimizer.state)
    rejected = torch.nn.Parameter(torch.zeros(2, 2))
    with torch.no_grad(), pytest.raises(ValueError, match="fraction"):
        optimizer.add_param_group(
            {
                "params": [rejected],
                "algorithm": "dion3",
                "fraction": 0.0,
            }
        )
    assert len(optimizer.param_groups) == original_group_count
    assert set(optimizer.state) == original_state_parameters

    with pytest.raises(ValueError, match="Muon parameter groups"):
        Dion3(
            [{"params": [rejected], "algorithm": "muon"}],
        )


def test_dion3_cuda_batched_entry_points_match_individual_calls() -> None:
    generator = torch.Generator(device="cuda").manual_seed(317)
    shapes = ((64, 32), (96, 48))
    gradients = [torch.randn(shape, generator=generator) for shape in shapes]
    reference_momenta = [torch.randn(shape, generator=generator) for shape in shapes]
    batched_momenta = [momentum.clone() for momentum in reference_momenta]
    reference_parameters = [torch.randn(shape, generator=generator) for shape in shapes]
    batched_parameters = [parameter.clone() for parameter in reference_parameters]
    selected_counts = [16, 24]
    reference_updates: list[Tensor] = []
    reference_indices: list[Tensor] = []
    for tensor_index in range(len(shapes)):
        selected_update, indices = select_dion3_rows_cuda(
            reference_momenta[tensor_index],
            gradients[tensor_index],
            0.25,
            0.95,
            selected_counts[tensor_index],
            reference_parameters[tensor_index],
            0.999,
        )
        reference_updates.append(selected_update)
        reference_indices.append(indices)

    batched_updates, batched_indices = select_dion3_rows_cuda_batch(
        batched_momenta,
        gradients,
        batched_parameters,
        selected_counts,
        0.95,
        0.999,
    )
    for tensor_index in range(len(shapes)):
        torch.testing.assert_close(
            batched_momenta[tensor_index],
            reference_momenta[tensor_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched_parameters[tensor_index],
            reference_parameters[tensor_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched_updates[tensor_index],
            reference_updates[tensor_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched_indices[tensor_index],
            reference_indices[tensor_index],
            rtol=0,
            atol=0,
        )

    reference_variances = [
        torch.zeros((shape[0], 1), dtype=torch.float32) for shape in shapes
    ]
    batched_variances = [variance.clone() for variance in reference_variances]
    normalized_reference_parameters = [
        parameter.clone() for parameter in reference_parameters
    ]
    normalized_batched_parameters = [
        parameter.clone() for parameter in reference_parameters
    ]
    adjusted_learning_rates = [0.01, 0.02]
    for tensor_index in range(len(shapes)):
        normalize_apply_rows_cuda(
            normalized_reference_parameters[tensor_index],
            reference_updates[tensor_index],
            reference_variances[tensor_index],
            reference_indices[tensor_index],
            0.95,
            1e-8,
            0.01,
            0.0,
            adjusted_learning_rates[tensor_index],
        )
    normalize_apply_rows_cuda_batch(
        normalized_batched_parameters,
        reference_updates,
        batched_variances,
        reference_indices,
        0.95,
        1e-8,
        0.01,
        0.0,
        adjusted_learning_rates,
    )
    for tensor_index in range(len(shapes)):
        torch.testing.assert_close(
            normalized_batched_parameters[tensor_index],
            normalized_reference_parameters[tensor_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched_variances[tensor_index],
            reference_variances[tensor_index],
            rtol=0,
            atol=0,
        )
