# Attention Kernel 作业框架

本仓库用于实现大模型推理中的 scaled dot-product attention kernel，分别覆盖
prefill 和 decode 两个阶段。作业重点是熟悉 Triton/NineToothed 的 kernel 编程和
性能分析；FlashAttention 是可选的优化方向。完整的接口约定、评分标准和报告要求见
[`homeworkmd/homework.tex`](homeworkmd/homework.tex)。

## 安装依赖

请在课程提供的 CUDA 容器中执行：

```bash
pip install -r requirements.txt
```

当前作业依赖固定为 Torch 2.13.0、NineToothed 0.26.0，并使用与之匹配的 Triton
版本。请先确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 快速运行

`torch` backend 是参考基线，不能直接作为自定义 kernel 提交：

```bash
python main.py --stage prefill --backends torch
python main.py --stage decode  --backends torch
```

实现 Triton 或 NineToothed 后，可以与 Torch 基线一起检查：

```bash
python main.py --stage prefill --backends torch triton
python main.py --stage decode  --backends torch triton
python main.py --stage prefill --backends torch ninetoothed
python main.py --stage decode  --backends torch ninetoothed
```

首次运行会包含编译和 autotune 时间；性能报告应只统计预热后的稳态 kernel 延迟。

## 可选 CUDA 路径

如果选择 CUDA C++，RTX 4090 使用 compute capability 8.9：

```bash
cd ops/cuda/kernels
TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace
cd ../..
python main.py --stage prefill --backends cuda
```

Triton、NineToothed 和 Torch 路径不要求先编译 CUDA 扩展。提交前请阅读
[`HONOR-CODE.md`](HONOR-CODE.md)，如使用公开资料或工具，请按作业文档记录来源。
