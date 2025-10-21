import torch
import math

def scaled_dot_product_attention_prefill(query, key, value, scale=None) -> torch.Tensor:
    # 获取 query 和 key 的长度（L, S）
    L, S = query.size(-2), key.size(-2)
    
    # 如果没有提供 scale 参数，默认使用 sqrt(d_k)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # 初始化一个零的 attention 偏置矩阵
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    # 计算注意力权重
    # query @ key.transpose(-2, -1) 计算的是 Query 和 Key 的点积
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 将偏置加到注意力权重上
    attn_weight += attn_bias
    
    # softmax**
    
    # 步骤 1: 计算指数
    exp_weights = torch.exp(attn_weight)  # 对每个元素取指数
    
    # 步骤 2: 对每一行求和，得到归一化因子
    sum_exp_weights = exp_weights.sum(dim=-1, keepdim=True)  # 对每一行求和，keepdim=True 保持维度一致
    
    # 步骤 3: 归一化，每个元素除以该行的总和
    softmax_weights = exp_weights / sum_exp_weights  # 每行的元素除以该行的总和
    
    # 计算最终的加权值
    output = softmax_weights @ value
    
    return output

def scaled_dot_product_attention_decode(query, key, value, past_key, past_value, scale=None) -> torch.Tensor:
    
    # 获取 query 和 key 的长度（L, S）
    L, S = query.size(-2), key.size(-2)
    
    # 如果没有提供 scale 参数，默认使用 sqrt(d_k)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # 将 past_key 和 past_value 与当前 key 和 value 拼接（如果有 past_key 和 past_value）
    if past_key is not None and past_value is not None:
        key = torch.cat([past_key, key], dim=-2)  # 在键的维度上拼接 past_key 和 key
        value = torch.cat([past_value, value], dim=-2)  # 在值的维度上拼接 past_value 和 value

    
    # 初始化一个零的 attention 偏置矩阵
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    # 计算注意力权重
    # query @ key.transpose(-2, -1) 计算的是 Query 和 Key 的点积
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 将偏置加到注意力权重上
    attn_weight += attn_bias
    
    # softmax**
    
    # 步骤 1: 计算指数
    exp_weights = torch.exp(attn_weight)  # 对每个元素取指数
    
    # 步骤 2: 对每一行求和，得到归一化因子
    sum_exp_weights = exp_weights.sum(dim=-1, keepdim=True)  # 对每一行求和，keepdim=True 保持维度一致
    
    # 步骤 3: 归一化，每个元素除以该行的总和
    softmax_weights = exp_weights / sum_exp_weights  # 每行的元素除以该行的总和
    
    # 计算最终的加权值
    output = softmax_weights @ value
    
    return output