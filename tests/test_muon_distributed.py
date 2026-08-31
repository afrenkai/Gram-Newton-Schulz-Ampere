import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as distributed_checkpoint
from torch import distributed
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel

from gram_newton_schulz_ampere import Dion3, Muon


@pytest.fixture(scope="module")
def distributed_device_mesh() -> Generator[DeviceMesh, None, None]:
    if "RANK" not in os.environ:
        pytest.skip("run with torchrun")
    distributed.init_process_group("gloo")
    world_size = distributed.get_world_size()
    device_mesh = init_device_mesh(
        "cpu",
        (world_size,),
        mesh_dim_names=("shard",),
    )
    try:
        yield device_mesh
    finally:
        distributed.destroy_process_group()


def test_distributed_muon_matches_single_rank(
    distributed_device_mesh: DeviceMesh,
) -> None:
    process_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    generator = torch.Generator().manual_seed(67)
    ddp_model = DistributedDataParallel(torch.nn.Linear(4, 4, bias=False))
    ddp_optimizer = Muon(
        ddp_model.parameters(),
        lr=0.02,
        momentum=0.0,
        nesterov=False,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed.group.WORLD,
    )
    ddp_inputs = torch.randn(3, 4, generator=generator) + process_rank
    ddp_model(ddp_inputs).square().mean().backward()
    ddp_optimizer.step()
    ddp_gathered = [
        torch.empty_like(ddp_model.module.weight) for rank_index in range(world_size)
    ]
    distributed.all_gather(ddp_gathered, ddp_model.module.weight)
    for gathered_parameter in ddp_gathered[1:]:
        torch.testing.assert_close(
            gathered_parameter,
            ddp_gathered[0],
            rtol=0,
            atol=0,
        )

    replicated_parameters = [
        torch.nn.Parameter(torch.randn(8, 4, generator=generator))
        for parameter_index in range(3)
    ]
    for parameter in replicated_parameters:
        parameter.grad = torch.randn(8, 4, generator=generator)
    replicated_optimizer = Muon(
        replicated_parameters,
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed.group.WORLD,
    )
    replicated_optimizer.step()
    for parameter in replicated_parameters:
        gathered = [torch.empty_like(parameter) for rank_index in range(world_size)]
        distributed.all_gather(gathered, parameter.detach())
        for candidate in gathered[1:]:
            torch.testing.assert_close(candidate, gathered[0], rtol=0, atol=0)

    for shard_dimension, shape in (
        (0, (5, 4)),
        (1, (4, 5)),
        (0, (1, 5)),
        (0, (5, 6)),
        (1, (6, 5)),
    ):
        full_parameter = torch.randn(shape, generator=generator)
        full_gradient = torch.randn(shape, generator=generator)
        reference = torch.nn.Parameter(full_parameter.clone())
        reference.grad = full_gradient.clone()
        Muon(
            [reference],
            lr=0.03,
            momentum=0.0,
            nesterov=False,
            adjust_lr=None,
            ns_use_kernels=False,
        ).step()

        parameter = torch.nn.Parameter(
            distribute_tensor(
                full_parameter,
                distributed_device_mesh,
                [Shard(shard_dimension)],
            )
        )
        parameter.grad = distribute_tensor(
            full_gradient,
            distributed_device_mesh,
            [Shard(shard_dimension)],
        )
        optimizer = Muon(
            [parameter],
            lr=0.03,
            momentum=0.0,
            nesterov=False,
            adjust_lr=None,
            ns_use_kernels=False,
            distributed_mesh=distributed_device_mesh,
        )
        optimizer.step()
        torch.testing.assert_close(
            parameter.full_tensor(),
            reference.detach(),
            rtol=0,
            atol=0,
        )
        momentum = optimizer.state[parameter]["momentum_buffer"]
        assert isinstance(momentum, DTensor)
        assert momentum.to_local().shape == parameter.to_local().shape

    regular_parameter = torch.nn.Parameter(torch.zeros(3, 2))
    sharded_parameter = torch.nn.Parameter(
        distribute_tensor(torch.zeros(4, 2), distributed_device_mesh, [Shard(0)])
    )
    regular_parameter.grad = torch.ones_like(regular_parameter)
    sharded_gradient = distribute_tensor(
        torch.ones(4, 2),
        distributed_device_mesh,
        [Shard(0)],
    )
    if process_rank == 0:
        sharded_parameter.grad = sharded_gradient
    mixed_optimizer = Muon(
        [regular_parameter, sharded_parameter],
        ns_use_kernels=False,
    )
    with pytest.raises(RuntimeError, match="gradient participation"):
        mixed_optimizer.step()

    if process_rank == 0:
        print("distributed Muon parity passed")


def test_distributed_dion3_matches_single_rank(
    distributed_device_mesh: DeviceMesh,
) -> None:
    process_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    generator = torch.Generator().manual_seed(113)
    replicated_parameters = [
        torch.nn.Parameter(torch.randn(8, 4, generator=generator))
        for parameter_index in range(3)
    ]
    for parameter in replicated_parameters:
        parameter.grad = torch.randn(8, 4, generator=generator)
    replicated_optimizer = Dion3(
        replicated_parameters,
        lr=0.05,
        fraction=0.5,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed.group.WORLD,
    )
    replicated_optimizer.step()
    for parameter in replicated_parameters:
        gathered = [torch.empty_like(parameter) for rank_index in range(world_size)]
        distributed.all_gather(gathered, parameter.detach())
        for candidate in gathered[1:]:
            torch.testing.assert_close(candidate, gathered[0], rtol=0, atol=0)

    full_parameter = torch.randn(6, 4, generator=generator)
    full_gradient = torch.randn(6, 4, generator=generator)
    reference = torch.nn.Parameter(full_parameter.clone())
    reference.grad = full_gradient.clone()
    Dion3(
        [reference],
        lr=0.03,
        fraction=1.0,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
        ns_use_kernels=False,
    ).step()

    parameter = torch.nn.Parameter(
        distribute_tensor(full_parameter, distributed_device_mesh, [Shard(0)])
    )
    parameter.grad = distribute_tensor(
        full_gradient,
        distributed_device_mesh,
        [Shard(0)],
    )
    optimizer = Dion3(
        [parameter],
        lr=0.03,
        fraction=1.0,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed_device_mesh,
    )
    optimizer.step()
    distributed_parameter = parameter.full_tensor()
    reference_update = full_parameter - reference.detach()
    distributed_update = full_parameter - distributed_parameter
    relative_difference = (
        distributed_update - reference_update
    ).norm() / reference_update.norm().clamp(min=1e-8)
    cosine = torch.nn.functional.cosine_similarity(
        distributed_update.flatten(),
        reference_update.flatten(),
        dim=0,
    )
    assert relative_difference < 0.2
    assert cosine > 0.98
    momentum = optimizer.state[parameter]["momentum_buffer"]
    variance = optimizer.state[parameter]["variance_neuron"]
    assert isinstance(momentum, DTensor)
    assert isinstance(variance, DTensor)
    assert variance.to_local().shape == (3, 1)

    model = torch.nn.Module()
    model.register_parameter("matrix", parameter)
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    resumed_parameter = torch.nn.Parameter(
        distribute_tensor(
            distributed_parameter.detach(),
            distributed_device_mesh,
            [Shard(0)],
        )
    )
    resumed_model = torch.nn.Module()
    resumed_model.register_parameter("matrix", resumed_parameter)
    resumed_optimizer = Dion3(
        [resumed_parameter],
        lr=0.03,
        fraction=1.0,
        momentum=0.5,
        muon_beta2=0.9,
        weight_decay=0.0,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed_device_mesh,
    )
    checkpoint_path = Path(tempfile.gettempdir()) / (
        f"ampere-dion3-dcp-{os.environ['MASTER_PORT']}"
    )
    if process_rank == 0 and checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    distributed.barrier()
    distributed_checkpoint.save(
        {"optimizer": optimizer_state},
        checkpoint_id=str(checkpoint_path),
    )
    loaded_optimizer_state = get_optimizer_state_dict(
        resumed_model,
        resumed_optimizer,
    )
    distributed_checkpoint.load(
        {"optimizer": loaded_optimizer_state},
        checkpoint_id=str(checkpoint_path),
    )
    set_optimizer_state_dict(
        resumed_model,
        resumed_optimizer,
        loaded_optimizer_state,
    )
    resumed_momentum = resumed_optimizer.state[resumed_parameter]["momentum_buffer"]
    resumed_variance = resumed_optimizer.state[resumed_parameter]["variance_neuron"]
    assert isinstance(resumed_momentum, DTensor)
    assert isinstance(resumed_variance, DTensor)
    torch.testing.assert_close(
        resumed_momentum.to_local(),
        momentum.to_local(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        resumed_variance.to_local(),
        variance.to_local(),
        rtol=0,
        atol=0,
    )
    distributed.barrier()
    if process_rank == 0:
        shutil.rmtree(checkpoint_path)

    mixed_full_parameters = (
        torch.randn(6, 4, generator=generator),
        torch.randn(5, 4, generator=generator),
    )
    mixed_parameters = [
        torch.nn.Parameter(
            distribute_tensor(
                mixed_full_parameter,
                distributed_device_mesh,
                [Shard(0)],
            )
        )
        for mixed_full_parameter in mixed_full_parameters
    ]
    for mixed_parameter in mixed_parameters:
        mixed_parameter.grad = torch.ones_like(mixed_parameter)
    mixed_optimizer = Dion3(
        mixed_parameters,
        fraction=1.0,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed_device_mesh,
    )
    mixed_optimizer.step()
    assert mixed_parameters[0].to_local().shape[0] == 3
    assert mixed_parameters[1].to_local().shape[0] in {2, 3}

    batched_parameters = [
        torch.nn.Parameter(torch.randn(2, 4, 3, generator=generator)),
        torch.nn.Parameter(torch.randn(2, 4, 5, generator=generator)),
    ]
    for batched_parameter in batched_parameters:
        batched_parameter.grad = torch.ones_like(batched_parameter)
    Dion3(
        batched_parameters,
        fraction=0.5,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed.group.WORLD,
    ).step()
    assert all(torch.isfinite(parameter).all() for parameter in batched_parameters)

    batched_full = torch.randn(2, 5, 4, generator=generator)
    batched_shard = torch.nn.Parameter(
        distribute_tensor(batched_full, distributed_device_mesh, [Shard(1)])
    )
    batched_shard.grad = torch.ones_like(batched_shard)
    Dion3(
        [batched_shard],
        fraction=0.5,
        adjust_lr=None,
        ns_use_kernels=False,
        distributed_mesh=distributed_device_mesh,
    ).step()
    assert torch.isfinite(batched_shard.to_local()).all()

    column_parameter = torch.nn.Parameter(
        distribute_tensor(full_parameter, distributed_device_mesh, [Shard(1)])
    )
    column_parameter.grad = distribute_tensor(
        full_gradient,
        distributed_device_mesh,
        [Shard(1)],
    )
    column_optimizer = Dion3(
        [column_parameter],
        fraction=0.5,
        ns_use_kernels=False,
        distributed_mesh=distributed_device_mesh,
    )
    with pytest.raises(NotImplementedError, match="row-sharded"):
        column_optimizer.step()

    if process_rank == 0:
        print("distributed Dion3 parity passed")
