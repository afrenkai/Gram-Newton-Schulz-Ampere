import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=[
        "batch_size",
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
    ],
)
@triton.jit
def baddbmm_kernel(
    C_ptr,
    A_ptr,
    B_ptr,
    D_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_db,
    stride_dm,
    stride_dn,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_ptrs = (
        A_ptr
        + batch_id * stride_ab
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    B_ptrs = (
        B_ptr
        + batch_id * stride_bb
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            A_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            B_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    C_ptrs = (
        C_ptr
        + batch_id * stride_cb
        + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    )
    c = tl.load(C_ptrs, mask=mask, other=0.0)

    c = c.to(tl.float32)
    out = alpha * accumulator + beta * c

    D_ptrs = (
        D_ptr
        + batch_id * stride_db
        + (offs_cm[:, None] * stride_dm + offs_cn[None, :] * stride_dn)
    )
    tl.store(D_ptrs, out.to(C_ptr.dtype.element_ty), mask=mask)


def triton_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    if left.shape[0] != right.shape[0] or left.shape[0] != accumulator.shape[0]:
        raise ValueError("Batch sizes must match")
    if left.shape[1] != accumulator.shape[1]:
        raise ValueError("M dimensions must match")
    if right.shape[2] != accumulator.shape[2]:
        raise ValueError("N dimensions must match")
    if left.shape[2] != right.shape[1]:
        raise ValueError("K dimensions must match")

    batch_size, rows, inner_dimension = left.shape
    columns = right.shape[2]
    output = torch.empty_like(accumulator)

    def launch_grid(metadata: dict[str, int]) -> tuple[int, int]:
        matrix_programs = triton.cdiv(
            rows,
            metadata["BLOCK_SIZE_M"],
        ) * triton.cdiv(columns, metadata["BLOCK_SIZE_N"])
        return matrix_programs, batch_size

    baddbmm_kernel[launch_grid](
        accumulator,
        left,
        right,
        output,
        alpha,
        beta,
        rows,
        columns,
        inner_dimension,
        accumulator.stride(0),
        accumulator.stride(1),
        accumulator.stride(2),
        left.stride(0),
        left.stride(1),
        left.stride(2),
        right.stride(0),
        right.stride(1),
        right.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        batch_size,
    )
    return output


class TritonBackend:
    def symmetric_matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def symmetric_batch_matrix_matrix_product(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor:
        return triton_baddbmm(
            accumulator,
            left,
            right,
            alpha=alpha,
            beta=beta,
        )

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return left @ right

    def matmul_add(
        self,
        left: Tensor,
        right: Tensor,
        accumulator: Tensor,
        beta: float,
    ) -> Tensor:
        return triton_baddbmm(accumulator, left, right, beta=beta)
