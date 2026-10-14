import math

import torch

import ops.ninetoothed.kernels.scaled_dot_product_attention_decode
import ops.ninetoothed.kernels.scaled_dot_product_attention_prefill

def scaled_dot_product_attention_prefill(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)

    ops.ninetoothed.kernels.scaled_dot_product_attention_prefill.kernel(q, k, v, scale, o)

    return o

def scaled_dot_product_attention_decode(q, k, v, past_k, past_v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)

    # This reference path concatenates the cache before launching the prefill
    # kernel. Students can replace it with a cache-aware kernel to avoid this
    # temporary allocation and measure the decode-specific trade-off.
    k_all = torch.cat((past_k, k), dim=-2)
    v_all = torch.cat((past_v, v), dim=-2)
    ops.ninetoothed.kernels.scaled_dot_product_attention_prefill.kernel(
        q, k_all, v_all, scale, o
    )

    return o
