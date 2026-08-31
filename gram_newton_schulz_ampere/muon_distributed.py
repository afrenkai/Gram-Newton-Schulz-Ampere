from collections.abc import Sequence

import torch
from torch import Tensor, distributed
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Shard

from .muon_types import (
    DistributedMesh,
    Orthogonalizer,
    ParameterLayout,
)


def local_tensor(tensor: Tensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def resolve_gradient(parameter: Tensor) -> Tensor:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("Cannot resolve a missing gradient")
    if isinstance(parameter, DTensor):
        if not isinstance(gradient, DTensor):
            raise TypeError("A DTensor parameter requires a DTensor gradient")
        if gradient.device_mesh != parameter.device_mesh:
            raise ValueError("Parameter and gradient must use the same DeviceMesh")
        if gradient.placements != parameter.placements:
            gradient = gradient.redistribute(
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
        return gradient.to_local()
    if isinstance(gradient, DTensor):
        raise TypeError("A regular Tensor parameter cannot use a DTensor gradient")
    return gradient


def configured_process_group(
    distributed_mesh: DistributedMesh,
) -> ProcessGroup | None:
    if isinstance(distributed_mesh, ProcessGroup):
        return distributed_mesh
    if isinstance(distributed_mesh, DeviceMesh):
        if distributed_mesh.ndim != 1:
            raise ValueError(
                "Pass a 1D optimizer submesh, not a multidimensional DeviceMesh"
            )
        if distributed_mesh.size() > 1:
            return distributed_mesh.get_group()
    return None


def parameter_layout(
    parameter: Tensor,
    distributed_mesh: DistributedMesh,
    flatten: bool,
) -> ParameterLayout:
    configured_group = configured_process_group(distributed_mesh)
    if not isinstance(parameter, DTensor):
        return ParameterLayout(configured_group, None, False)

    active_shards: list[tuple[int, Shard]] = []
    for mesh_dimension, placement in enumerate(parameter.placements):
        if placement.is_partial():
            raise NotImplementedError("Partial DTensor parameters are not supported")
        if (
            isinstance(placement, Shard)
            and parameter.device_mesh.size(mesh_dimension) > 1
        ):
            active_shards.append((mesh_dimension, placement))
    if len(active_shards) > 1:
        raise NotImplementedError(
            "Muon supports one active DTensor shard dimension; combined FSDP2 and TP is not supported"
        )
    if not active_shards:
        return ParameterLayout(configured_group, None, False)

    mesh_dimension, shard = active_shards[0]
    shard_group = parameter.device_mesh.get_group(mesh_dimension)
    if configured_group is not None and configured_group != shard_group:
        raise ValueError(
            "The optimizer process group must match the parameter shard mesh axis"
        )

    matrix_dimensions = {parameter.ndim - 2, parameter.ndim - 1}
    batch_sharded = not flatten and shard.dim not in matrix_dimensions
    if flatten and shard.dim != 0:
        raise NotImplementedError(
            "Flattened tensors can only be sharded on output dimension 0"
        )
    return ParameterLayout(shard_group, shard.dim, batch_sharded)


def validate_gradient_participation(
    parameters: Sequence[Tensor],
    distributed_mesh: DistributedMesh,
    flatten: bool,
) -> None:
    participation_groups: dict[
        tuple[ProcessGroup, torch.device],
        list[bool],
    ] = {}
    for parameter in parameters:
        layout = parameter_layout(parameter, distributed_mesh, flatten)
        if layout.process_group is None:
            continue
        group_key = (layout.process_group, local_tensor(parameter).device)
        if group_key not in participation_groups:
            participation_groups[group_key] = []
        participation_groups[group_key].append(parameter.grad is not None)

    for group_key, participation in participation_groups.items():
        process_group, local_device = group_key
        if distributed.get_world_size(process_group) == 1:
            continue
        present = torch.tensor(
            participation,
            dtype=torch.int32,
            device=local_device,
        )
        minimum_present = present.clone()
        maximum_present = present.clone()
        distributed.all_reduce(
            minimum_present,
            op=distributed.ReduceOp.MIN,
            group=process_group,
        )
        distributed.all_reduce(
            maximum_present,
            op=distributed.ReduceOp.MAX,
            group=process_group,
        )
        if not torch.equal(minimum_present, maximum_present):
            raise RuntimeError(
                "Distributed orthogonal optimizers require the same gradient "
                "participation on every rank"
            )


def orthogonalize_parameter_updates(
    parameters: Sequence[Tensor],
    local_updates: Sequence[Tensor],
    global_matrix_shapes: Sequence[tuple[int, ...]],
    orthogonalizer: Orthogonalizer,
    distributed_mesh: DistributedMesh,
    flatten: bool,
) -> list[Tensor]:
    if len(parameters) != len(local_updates) or len(parameters) != len(
        global_matrix_shapes
    ):
        raise ValueError(
            "Parameters, updates, and global shapes must have equal lengths"
        )
    if not parameters:
        return []

    shape_groups: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            torch.dtype,
            torch.device,
            int | None,
            bool,
        ],
        list[int],
    ] = {}
    layouts: list[ParameterLayout] = []
    for parameter_index, parameter in enumerate(parameters):
        layout = parameter_layout(parameter, distributed_mesh, flatten)
        layouts.append(layout)
        shape_key = (
            tuple(parameter.shape),
            global_matrix_shapes[parameter_index],
            local_updates[parameter_index].dtype,
            local_updates[parameter_index].device,
            layout.sharded_tensor_dimension,
            layout.batch_sharded,
        )
        if shape_key not in shape_groups:
            shape_groups[shape_key] = []
        shape_groups[shape_key].append(parameter_index)

    outputs = [torch.empty_like(update) for update in local_updates]
    for matrix_indices in shape_groups.values():
        first_index = matrix_indices[0]
        layout = layouts[first_index]
        matrices = [local_updates[matrix_index] for matrix_index in matrix_indices]
        for matrix_index in matrix_indices:
            if layouts[matrix_index] != layout:
                raise ValueError(
                    "One shape bucket cannot span different optimizer mesh layouts"
                )

        if layout.batch_sharded:
            bucket_outputs = orthogonalize_local_matrices(matrices, orthogonalizer)
        elif layout.sharded_tensor_dimension is not None:
            logical_shard_dimension = layout.sharded_tensor_dimension
            if flatten:
                logical_shard_dimension = 0
            bucket_outputs = orthogonalize_sharded_matrices(
                matrices,
                global_matrix_shapes[first_index],
                logical_shard_dimension,
                orthogonalizer,
                layout.process_group,
            )
        else:
            bucket_outputs = orthogonalize_replicated_matrices(
                matrices,
                orthogonalizer,
                layout.process_group,
            )

        for output_index, matrix_index in enumerate(matrix_indices):
            outputs[matrix_index] = bucket_outputs[output_index]
    return outputs


def orthogonalize_local_matrices(
    matrices: Sequence[Tensor],
    orthogonalizer: Orthogonalizer,
) -> list[Tensor]:
    stacked_outputs = orthogonalizer(torch.stack(list(matrices)))
    return [stacked_outputs[matrix_index] for matrix_index in range(len(matrices))]


def orthogonalize_replicated_matrices(
    matrices: Sequence[Tensor],
    orthogonalizer: Orthogonalizer,
    process_group: ProcessGroup | None,
) -> list[Tensor]:
    if process_group is None or distributed.get_world_size(process_group) == 1:
        return orthogonalize_local_matrices(matrices, orthogonalizer)

    world_size = distributed.get_world_size(process_group)
    process_rank = distributed.get_rank(process_group)
    matrix_count = len(matrices)
    local_matrix_count = (matrix_count + world_size - 1) // world_size
    padded_matrix_count = local_matrix_count * world_size
    padded_matrices = list(matrices)
    while len(padded_matrices) < padded_matrix_count:
        padded_matrices.append(torch.zeros_like(matrices[0]))

    local_start = process_rank * local_matrix_count
    local_end = local_start + local_matrix_count
    local_outputs = orthogonalizer(
        torch.stack(padded_matrices[local_start:local_end])
    ).contiguous()
    gathered_chunks = [
        torch.empty_like(local_outputs) for rank_index in range(world_size)
    ]
    distributed.all_gather(
        gathered_chunks,
        local_outputs,
        group=process_group,
    )
    gathered_outputs = torch.cat(gathered_chunks, dim=0)[:matrix_count]
    return [gathered_outputs[matrix_index] for matrix_index in range(matrix_count)]


def orthogonalize_sharded_matrices(
    matrices: Sequence[Tensor],
    global_matrix_shape: tuple[int, ...],
    sharded_tensor_dimension: int,
    orthogonalizer: Orthogonalizer,
    process_group: ProcessGroup | None,
) -> list[Tensor]:
    if process_group is None:
        raise ValueError("A sharded DTensor update requires a process group")

    world_size = distributed.get_world_size(process_group)
    matrix_count = len(matrices)
    padded_matrix_count = ((matrix_count + world_size - 1) // world_size) * world_size
    matrices_per_rank = padded_matrix_count // world_size
    padded_matrices = list(matrices)
    while len(padded_matrices) < padded_matrix_count:
        padded_matrices.append(torch.zeros_like(matrices[0]))

    communication_dimension = sharded_tensor_dimension - matrices[0].ndim
    global_communication_size = global_matrix_shape[sharded_tensor_dimension]
    padded_local_size = (global_communication_size + world_size - 1) // world_size
    original_local_size = matrices[0].shape[sharded_tensor_dimension]
    if original_local_size > padded_local_size:
        raise RuntimeError(
            "The local shard exceeds the expected contiguous DTensor shard size"
        )
    if original_local_size < padded_local_size:
        padded_matrices = [
            pad_matrix_dimension(
                matrix,
                sharded_tensor_dimension,
                padded_local_size,
            )
            for matrix in padded_matrices
        ]

    input_chunks = list(
        torch.stack(padded_matrices)
        .unflatten(0, (world_size, matrices_per_rank))
        .unbind(0)
    )
    assembled_chunks = [torch.empty_like(chunk) for chunk in input_chunks]
    distributed.all_to_all(
        assembled_chunks,
        input_chunks,
        group=process_group,
    )
    full_matrices = torch.cat(assembled_chunks, dim=communication_dimension)
    full_outputs = orthogonalizer(full_matrices)

    split_chunks = [
        chunk.contiguous()
        for chunk in torch.tensor_split(
            full_outputs,
            world_size,
            dim=communication_dimension,
        )
    ]
    returned_chunks = [torch.empty_like(chunk) for chunk in split_chunks]
    distributed.all_to_all(
        returned_chunks,
        split_chunks,
        group=process_group,
    )
    returned_outputs = (
        torch.stack(returned_chunks)
        .narrow(communication_dimension, 0, original_local_size)
        .contiguous()
        .flatten(0, 1)[:matrix_count]
    )
    return [returned_outputs[matrix_index] for matrix_index in range(matrix_count)]


def pad_matrix_dimension(
    matrix: Tensor,
    tensor_dimension: int,
    padded_size: int,
) -> Tensor:
    padding_shape = list(matrix.shape)
    padding_shape[tensor_dimension] = padded_size - matrix.shape[tensor_dimension]
    padding = torch.zeros(
        padding_shape,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    return torch.cat((matrix, padding), dim=tensor_dimension)
