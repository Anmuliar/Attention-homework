import math

import torch


def scaled_dot_product_attention_prefill(query, key, value, scale=None) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = query @ key.transpose(-2, -1) * scale_factor
    return torch.softmax(scores, dim=-1) @ value


def scaled_dot_product_attention_decode(
    query, key, value, past_key, past_value, scale=None
) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if past_key is not None and past_value is not None:
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    # The cache is part of the key/value sequence in decode, so the score
    # matrix has the full cached length on its last dimension.
    scores = query @ key.transpose(-2, -1) * scale_factor
    return torch.softmax(scores, dim=-1) @ value
