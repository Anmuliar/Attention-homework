#include <torch/extension.h>

// CUDA forward declarations
using at::IntArrayRef;
using torch::Tensor;
using namespace torch::indexing;

void scaled_dot_product_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_k,
    const torch::Tensor& past_v,
    torch::Tensor& output
);

void scaled_dot_product_attention_prefill(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& output
);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x)                                                     \
    CHECK_CPU(x);                                                              \
    CHECK_CONTIGUOUS(x)


void scaled_dot_product_attention_prefill_cuda_forward(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor& output
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    scaled_dot_product_attention_prefill(
        q, k, v, output
    );
}

void scaled_dot_product_attention_decode_cuda_forward(
    const Tensor& q, const Tensor& k, const Tensor& v, 
    const Tensor& past_k, const Tensor& past_v, Tensor& output
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(past_k);
    CHECK_INPUT(past_v);
    scaled_dot_product_attention_decode(
        q, k, v, past_k, past_v, output
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_dot_product_attention_decode", &scaled_dot_product_attention_decode_cuda_forward,
          "scaled_dot_product_attention_decode (CUDA, Attention)")
     .def("scaled_dot_product_attention_prefill", &scaled_dot_product_attention_prefill_cuda_forward,
          "scaled_dot_product_attention_prefill (CUDA, Attention)");
}