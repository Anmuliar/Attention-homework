from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# The course benchmark uses RTX 4090 (compute capability 8.9).  Override this
# from the shell when compiling for another GPU, for example:
# TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
print(torch_lib_path)
if False:  # Debug
    extra_compile_args = {
        "nvcc": ["-G", "-g"],
        "cxx": ["-g", "-Og"],
    }
else:  # Release
    extra_compile_args = {
        "nvcc": [
            "-g",
            "--use_fast_math",
            "-lineinfo",
        ],
        "cxx": ["-g"],
    }

setup(
    name="cuda_attention",
    ext_modules=[
        CUDAExtension(
            name="cuda_attention",
            sources=[
                "operators.cpp",
                "scaled_dot_product_attention_decode.cu",
                "scaled_dot_product_attention_prefill.cu",
            ],
            library_dirs=[torch_lib_path], 
            libraries=["c10", "torch", "torch_cpu", "torch_cuda"],  
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
