
import torch
import triton
import math

import ops.triton.kernels.scaled_dot_product_attention_decode
import ops.triton.kernels.scaled_dot_product_attention_prefill

def scaled_dot_product_attention_prefill(q, k, v, scale=None):
    batch_size, num_heads, seq_len_q, emb_dim = q.shape
    _, _, seq_len_k_v, _ = k.shape

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    o = torch.empty_like(q)

    def grid(meta):
        return (
            triton.cdiv(seq_len_q, meta["BLOCK_SIZE_M"]),
            num_heads,
            batch_size,
        )
    # TODO: 完成对应kernel的调用
    assert False, "This function is not implemented yet."
    # ops.triton.kernels.scaled_dot_product_attention_prefill.kernel[grid](
    #     q,
    #     k,
    #     v,
    #     o,
    #     *q.stride(),
    #     *k.stride(),
    #     *v.stride(),
    #     *o.stride(),
    #     scale=scale,
    #     seq_len_q=seq_len_q,
    #     seq_len_k_v=seq_len_k_v,
    #     EMB_DIM=emb_dim,
    # )

    return o

def scaled_dot_product_attention_decode(q, k, v, past_k, past_v, scale=None):
    batch_size, num_heads, seq_len_q, emb_dim = q.shape
    _, _, seq_len_k_v, _ = k.shape

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    o = torch.empty_like(q)

    def grid(meta):
        return (
            triton.cdiv(seq_len_q, meta["BLOCK_SIZE_M"]),
            num_heads,
            batch_size,
        )
    # TODO: 完成对应kernel的调用
    assert False, "This function is not implemented yet."
    # ops.triton.kernels.scaled_dot_product_attention_decode.kernel[grid](
    #     q,
    #     k,
    #     v,
    #     past_k,
    #     past_v,
    #     o,
    #     *q.stride(),
    #     *k.stride(),
    #     *v.stride(),
    #     *past_k.stride(),
    #     *past_v.stride(),
    #     *o.stride(),
    #     scale=scale,
    #     seq_len_q=seq_len_q,
    #     seq_len_k_v=seq_len_k_v,
    #     EMB_DIM=emb_dim,
    # )

    return o
