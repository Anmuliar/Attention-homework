from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from transformers.models.llama.modeling_llama import repeat_kv
import argparse

import ops.ninetoothed.torch
import ops.triton.torch
import ops.base.impl
import cuda_attention

parser = argparse.ArgumentParser(description="Run attention test with different backends and stages.")

parser.add_argument(
    '--stage',
    choices=['prefill', 'decode'],
    default="prefill",
    help="Choose the stage: 'prefill' or 'decode'."
)

parser.add_argument(
    '--backends',
    choices=['ninetoothed', 'cuda', 'triton', 'torch'],
    nargs='+',  # 支持传入多个选项
    default="torch",
    help="Choose one or more languages: 'ninetoothed', 'cuda', 'triton'."
)

args = parser.parse_args()

# A typical Attention example here, we do not need to implement the whole 
# attention module, only the scale_dot_product part
class Attention(nn.Module):
    scaled_dot_product_attention = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value,
        cache_position,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos_table, sin_table = position_embeddings
        sin_table = sin_table[0]
        cos_table = cos_table[0]

        query_states = type(self).rotary_position_embedding(
            query_states, sin_table, cos_table
        )
        key_states = type(self).rotary_position_embedding(
            key_states, sin_table, cos_table
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # In prefill stage there's no past_key_value as kv-cache
        # In decode stage we could use kv-cache to avoid kv recompute
        # mentioned in L57-L58, but we need to update the kv states.
        # In this homework you don't need to manage the kv, it will 
        # directly passes as parameters for your scaled_dot_product_attention kernel
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_table,
                "cos": cos_table,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # scaled_dot_product_attention kernel is our target in this homework,
        # we suggest you implement two kinds of attention for 
        # decode and prefill senarioes to achieve better performance.
        attn_output = type(self).scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )
        
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


@contextmanager
def scaled_dot_product_attention_backend(backend_name, stage = args.stage):
    _prev_impl = Attention.scaled_dot_product_attention

    if backend_name == "ninetoothed":
        if stage == "decode":
            impl = ops.ninetoothed.torch.scaled_dot_product_attention_decode
        elif stage == "prefill":
            impl = ops.ninetoothed.torch.scaled_dot_product_attention_prefill
    elif backend_name == "triton":
        if stage == "decode":
            impl = ops.triton.torch.scaled_dot_product_attention_decode
        elif stage == "prefill":
            impl = ops.triton.torch.scaled_dot_product_attention_prefill
    elif backend_name == "torch":
        if stage == "decode":
            impl = F.scaled_dot_product_attention
        elif stage == "prefill":
            impl = F.scaled_dot_product_attention
    elif backend_name == "cuda":
        if stage == "decode":
            impl = cuda_attention.scaled_dot_product_attention_decode
        elif stage == "prefill":
            impl = cuda_attention.scaled_dot_product_attention_prefill
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    Attention.scaled_dot_product_attention = impl
    
    try:
        yield
    finally:
        Attention.scaled_dot_product_attention = _prev_impl

def generate_prefill_input(batch_size, num_heads, seq_len, head_dim, dtype, device):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    o = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    return (q, k, v), o
    
def generate_decode_input(batch_size, num_heads, seq_len, head_dim, dtype, device):
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
    past_k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    past_v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    o = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)

    return (q, k, v, past_k, past_v), o

if __name__ == "__main__":
    
    print(f"Stage: {args.stage}")
    print(f"Backends to test: {args.backends}")
    
    # 根据 stage 执行不同阶段逻辑
    if args.stage == 'prefill':
        print("Running PREFILL stage...\n")
    else:
        print("Running DECODE stage...\n")
    torch.manual_seed(0)

    dtype = torch.float16
    device = "cuda"

    if args.stage == "prefill":
        inputs, o = generate_prefill_input(2, 8, 1024, 64, dtype, device)
        base_output = ops.base.impl.scaled_dot_product_attention_prefill(*inputs)
    if args.stage == "decode":
        inputs, o = generate_decode_input(2, 8, 1024, 64, dtype, device)
        base_output = ops.base.impl.scaled_dot_product_attention_decode(*inputs)
    
    if "torch" in args.backends:
        if args.stage == "prefill":
            torch_output = F.scaled_dot_product_attention(*inputs)
        if args.stage == "decode":
            q, k, v, past_k, past_v = inputs
            k_new = torch.cat([past_k, k], dim=-2)
            v_new = torch.cat([past_v, v], dim=-2)
            torch_output = F.scaled_dot_product_attention(q, k_new, v_new)
        if torch.allclose(torch_output, base_output, atol=0.01):
            print("✅ Torch and benchmark match.")  
        else:
            print("❌ Torch and benchmark differ.")
        
    if "triton" in args.backends:
        if args.stage == "prefill":
            triton_output = ops.triton.torch.scaled_dot_product_attention_prefill(*inputs)
        if args.stage == "decode":
            triton_output = ops.triton.torch.scaled_dot_product_attention_decode(*inputs)
        if torch.allclose(triton_output, base_output, atol=0.01):
            print("✅ Triton and benchmark match.")  
        else:
            print("❌ Triton and benchmark differ.")
            
    if "ninetoothed" in args.backends:
        if args.stage == "prefill":
            ninetoothed_output = ops.ninetoothed.torch.scaled_dot_product_attention_prefill(*inputs)
        if args.stage == "decode":
            ninetoothed_output = ops.ninetoothed.torch.scaled_dot_product_attention_decode(*inputs)
        if torch.allclose(ninetoothed_output, base_output, atol=0.01):
            print("✅ Ninetoothed and benchmark match.")  
        else:
            print("❌ Ninetoothed and benchmark differ.")
    
    if "cuda" in args.backends:
        if args.stage == "prefill":
            cuda_attention.scaled_dot_product_attention_prefill(*inputs, output=o)
        if args.stage == "decode":
            cuda_attention.scaled_dot_product_attention_decode(*inputs, output=o)
        if torch.allclose(o, base_output, atol=0.01):
            print("✅ Cuda and benchmark match.")  
        else:
            print("❌ Cuda and benchmark differ.")

    def benchmark(seq_len, provider):
        batch_size, num_heads, emb_dim = 4, 32, 64
        shape = (batch_size, num_heads, seq_len, emb_dim)
        dtype = torch.float16
        device = "cuda"

        if args.stage == "prefill":
            inputs, o = generate_prefill_input(batch_size, num_heads, seq_len, emb_dim, dtype, device)
        if args.stage == "decode":
            inputs, o = generate_decode_input(batch_size, num_heads, seq_len, emb_dim, dtype, device)

        if provider == "ninetoothed":
            if args.stage == "prefill":
                ms = triton.testing.do_bench(
                    lambda: ops.ninetoothed.torch.scaled_dot_product_attention_prefill(*inputs)
                )
            if args.stage == "decode":
                ms = triton.testing.do_bench(
                    lambda: ops.ninetoothed.torch.scaled_dot_product_attention_decode(*inputs)
                )
        elif provider == "torch":
            if args.stage == "prefill":
                ms = triton.testing.do_bench(
                    lambda: F.scaled_dot_product_attention(*inputs)
                )
            if args.stage == "decode":
                k_new = torch.cat([past_k, k], dim=-2)
                v_new = torch.cat([past_v, v], dim=-2)
                ms = triton.testing.do_bench(
                    lambda: F.scaled_dot_product_attention(q, k_new, v_new)
                )
        elif provider == "triton":
            if args.stage == "prefill":
                ms = triton.testing.do_bench(
                    lambda: ops.triton.torch.scaled_dot_product_attention_prefill(*inputs)
                )
            if args.stage == "decode":
                ms = triton.testing.do_bench(
                    lambda: ops.triton.torch.scaled_dot_product_attention_decode(*inputs)
                )
        elif provider == "cuda":
            if args.stage == "prefill":
                ms = triton.testing.do_bench(
                    lambda: cuda_attention.scaled_dot_product_attention_prefill(*inputs, output)
                )
            if args.stage == "decode":
                ms = triton.testing.do_bench(
                    lambda: cuda_attention.scaled_dot_product_attention_decode(*inputs, output)
                )
        elif provider == "benchmark":
            if args.stage == "prefill":
                ms = triton.testing.do_bench(
                    lambda: ops.base.impl.scaled_dot_product_attention_prefill(*inputs)
                )
            if args.stage == "decode":
                ms = triton.testing.do_bench(
                    lambda: ops.base.impl.scaled_dot_product_attention_decode(*inputs)
                )
        return ms

    seq_len_values = [2**i for i in range(11)]  
    providers = args.backends
    print(type(providers), providers)
    
    for seq_len in seq_len_values:
        for provider in providers:
            print(f"Running benchmark for seq_len={seq_len} with provider={provider}...")
            ms = benchmark(seq_len, provider)
            print(f"Time for seq_len {seq_len} with {provider}: {ms} ms\n")

