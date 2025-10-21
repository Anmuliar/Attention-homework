#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#define THREADS_PER_BLOCK 128

template <typename T>
__global__ void scaled_dot_product_attention_decode_kernel(
    /*TODO*/
) {
    /*TODO*/
}

/**
 * @brief Computes attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, query_heads, 1, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, kv_heads, 1, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, kv_heads, 1, head_dim]
 * @param[in] past_k K-cache tensor of shape [batch_size, kv_heads, seq_len, head_dim]
 * @param[in] past_v V-cache tensor of shape [batch_size, kv_heads, seq_len, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, query_heads, 1, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] seq_len Target sequence length 
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 */

template <typename T>
void scaled_dot_product_attention_decode_wrapper(
    cudaStream_t stream,
    const T* h_q, const T* h_k,
    const T* h_v, const T* past_k, const T* past_v,
    T* h_o, const float scale,
    const int batch_size, const int seq_len,
    const int query_heads, const int kv_heads, const int head_dim) {       
    
    /* TODO
        Example:

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(batch_size * num_heads);
        size_t smem_size = (2 * qk_dim + 2 * v_dim) * sizeof(T) + qk_dim * (v_dim + 1) * sizeof(float);

        scaled_dot_product_attention_prefill_kernel<<<grid, block, smem_size, stream>>>(
        ......
        );
    */

}

// *********************************************************************

// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)

// DO NOT MODIFY THIS SECTION

// *********************************************************************

void scaled_dot_product_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_k,
    const torch::Tensor& past_v,
    torch::Tensor& output
) {
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(past_k.is_contiguous(), "past_k must be contiguous");
    TORCH_CHECK(past_v.is_contiguous(), "past_v must be contiguous");

    auto batch_size = q.size(0);
    auto seq_len = past_k.size(2);
    auto query_heads = q.size(1);
    auto kv_heads = v.size(1);
    auto head_dim = q.size(3);
    float scale = 1 / sqrt(float(head_dim));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "scaled_dot_product_attention_decode", ([&] {
            scaled_dot_product_attention_decode_wrapper<scalar_t>(
                at::cuda::getCurrentCUDAStream(), 
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                past_k.data_ptr<scalar_t>(),
                past_v.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                scale,
                batch_size,
                seq_len,
                query_heads,
                kv_heads,
                head_dim);
        }));
}