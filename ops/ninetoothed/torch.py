import math

import torch

import ops.ninetoothed.kernels.scaled_dot_product_attention_decode
import ops.ninetoothed.kernels.scaled_dot_product_attention_prefill

import torch

def scaled_dot_product_attention_prefill(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)

    # TODO: 完成对应kernel的调用
    assert False, "This function is not implemented yet."
    # ops.ninetoothed.kernels.scaled_dot_product_attention_prefill.kernel(q, k, v, scale, o)

    return o

def scaled_dot_product_attention_decode(q, k, v, past_k, past_v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)
    
    # TODO: 完成对应kernel的调用
    assert False, "This function is not implemented yet."
    # ops.ninetoothed.kernels.scaled_dot_product_attention_decode.kernel(q, k, v, past_k, past_v, scale, o)

    return o