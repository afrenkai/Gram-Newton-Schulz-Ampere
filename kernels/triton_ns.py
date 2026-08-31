import torch
import triton
import triton.language as tl


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
    key=["M", "N", "K"],
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # unreachable btw
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_ptrs = A_ptr + batch_id * stride_ab + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + batch_id * stride_bb + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
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

    C_ptrs = C_ptr + batch_id * stride_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c = tl.load(C_ptrs, mask=mask, other=0.0)

    c = c.to(tl.float32)
    out = alpha * accumulator + beta * c

    D_ptrs = D_ptr + batch_id * stride_db + (offs_cm[:, None] * stride_dm + offs_cn[None, :] * stride_dn)
    tl.store(D_ptrs, out.to(C_ptr.dtype.element_ty), mask=mask)


def triton_baddbmm(C, A, B, alpha=1.0, beta=1.0):
    assert A.shape[0] == B.shape[0] == C.shape[0], "Batch sizes must match"
    assert A.shape[1] == C.shape[1], "M dimension must match"
    assert B.shape[2] == C.shape[2], "N dimension must match"
    assert A.shape[2] == B.shape[1], "K dimension must match"

    batch, M, K = A.shape
    _, _, N = B.shape

    D = torch.empty_like(C)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch,
    )

    baddbmm_kernel[grid](
        C,
        A,
        B,
        D,
        alpha,
        beta,
        M,
        N,
        K,
        C.stride(0),
        C.stride(1),
        C.stride(2),
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        D.stride(0),
        D.stride(1),
        D.stride(2),
    )

    return D
