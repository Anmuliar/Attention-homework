
# 项目使用说明

## 安装依赖

首先，安装所需的依赖项：

```bash
pip install -r requirements.txt
```

推荐使用cuda12.8+torch2.9

## CUDA 测试

完成 CUDA kernel 编写后，请按照以下步骤安装并测试 CUDA 版本的实现：

1. 进入 `/ops/cuda/kernels` 目录并运行以下命令来安装 kernel：

   ```bash
   python setup.py install
   ```
    在其他架构上进行实验需要修改os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"为其他值。

2. 使用以下命令运行测试，选择 `prefill` 或 `decode` 阶段，并指定使用的后端为 CUDA：

   ```bash
   python --stage [prefill/decode] --backends cuda
   ```

## Triton 测试

使用以下命令测试 Triton 后端的基础结果。选择 `prefill` 或 `decode` 阶段，并指定使用的后端为 Triton：

```bash
python --stage [prefill/decode] --backends triton
```

## NineToothed测试

使用以下命令测试 NineToothed 后端的基础结果。选择 `prefill` 或 `decode` 阶段，并指定使用的后端为NineToothed：

```bash
python --stage [prefill/decode] --backends ninetoothed
```

---

### 说明：

* `--stage` 参数：指定执行阶段，可以是 `prefill` 或 `decode`，分别表示预填充阶段和解码阶段。
* `--backends` 参数：选择使用的后端，支持 `cuda`、`triton` 和 `ninetoothed`选择多个可以同时进行测试。

确保在执行之前已经正确配置并安装相应的后端（CUDA、Triton 或 NineToothed）,有任何问题请联系助教邮箱wcz24@mails.tsinghua.edu.cn。

