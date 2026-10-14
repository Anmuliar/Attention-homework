import itertools

import triton
import triton.language as tl


# This reference kernel keeps only one query tile and one key/value tile in
# registers. The running maximum and normalizer implement online softmax, so
# the L x S score matrix never has to be materialized in global memory.
@triton.autotune(
    configs=tuple(
        triton.Config(
            {"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": block_size_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, num_stages, num_warps in itertools.product(
            (32, 64, 128), (32, 64, 128), (2, 3), (4, 8)
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
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, EMB_DIM)
    q_mask = offs_m < seq_len_q
    d_mask = offs_k < EMB_DIM

    q_base = q_ptr + pid_z * q_stride_z + pid_h * q_stride_h
    q = tl.load(
        q_base + offs_m[:, None] * q_stride_m + offs_k[None, :] * q_stride_k,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    # exp2 is used below, hence the conversion from exp to exp2 scale.
    q = (q * scale * 1.44269504089).to(q_ptr.type.element_ty)

    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_M,), 1.0, dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)

    k_base = k_ptr + pid_z * k_stride_z + pid_h * k_stride_h
    v_base = v_ptr + pid_z * v_stride_z + pid_h * v_stride_h
    for start_n in range(0, tl.cdiv(seq_len_k_v, BLOCK_SIZE_N)):
        offs_n = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < seq_len_k_v

        k = tl.load(
            k_base + offs_n[:, None] * k_stride_n + offs_k[None, :] * k_stride_k,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * v_stride_k + offs_k[None, :] * v_stride_n,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_ptr.type.element_ty), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc = acc / l_i[:, None]
    o_base = o_ptr + pid_z * o_stride_z + pid_h * o_stride_h
    tl.store(
        o_base + offs_m[:, None] * o_stride_m + offs_k[None, :] * o_stride_n,
        acc.to(o_ptr.type.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )
