import itertools

import triton
import triton.language as tl


@triton.autotune(
    configs=tuple(
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (32, 64, 128, 256), (32, 64, 128), (2, 3, 4, 5), (4, 8)
        )
    ),
    key=["EMB_DIM"],
)
@triton.jit
def kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_stride_z,
    q_stride_h,
    q_stride_m,
    q_stride_k,
    k_stride_z,
    k_stride_h,
    k_stride_n,
    k_stride_k,
    v_stride_z,
    v_stride_h,
    v_stride_k,
    v_stride_n,
    o_stride_z,
    o_stride_h,
    o_stride_m,
    o_stride_n,
    scale,
    seq_len_q,
    seq_len_k_v,
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    #TODO: 实现kernel
    assert False, "This function is not implemented yet."