**简体中文** | [English](README.en.md)

<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>面向大语言模型、由规范驱动的 GPU 算子库——帮助 AI Agent 构建、评估和优化算子</strong></p>
  <p>基于 <a href="https://github.com/tile-ai/tilelang">TileLang</a> 构建</p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/tileops/manifest"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/benchmarks"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
  </p>
  <p>
    <a href="#安装"><b>安装</b></a> |
    <a href="#快速开始"><b>快速开始</b></a> |
    <a href="#文档"><b>文档</b></a>
  </p>
</div>

> **项目状态**：TileOPs 正在积极开发中，API 可能发生变化。

## 首届开源英才夏令营

夏令营学员应使用 `summer-camp-2026` 分支，并遵循[算子迁移指南](docs/summer-camp/README.md)。该指南规定了算子认领、Manifest/实现双 PR 流程、MetaX GPU 验证、Benchmark、Roofline 证据和验收要求；可认领课题见[候选算子清单](docs/summer-camp/TileKernels-MACA-待迁移算子盘点.md)。

推荐通过[模力方舟算力市场](https://ai.gitee.com/compute)租用在线 MetaX C500 算力，并选择夏令营专属镜像：`PyTorch-Agent / 2.8.0 / Python 3.12 / MACA 3.7.1.5`。具体使用方法和环境自检要求见[详细中文指南](docs/summer-camp/README.zh-CN.md)。

## 概述

TileOPs 是一个基于 [TileLang](https://github.com/tile-ai/tilelang)、面向大语言模型训练和推理的 GPU 算子库。除了持续提供可用于生产的算子，TileOPs 还探索一种**规范驱动的开发模式**：AI Agent 可以读取声明式算子规范、生成 Kernel 实现，并依据硬件理论性能上限进行评估，同时尽量减少人工脚手架。

### 架构

每个算子都严格分为两个层次：

- **Op**（L2）——无状态 Python 入口，负责参数校验、dtype 转换和内存布局，并兼容 CUDA Graph 与 `torch.compile`。
- **Kernel**（L1）——TileLang GPU 实现，包含针对具体硬件的优化策略。上游 TileOPs 的 Kernel 按 NVIDIA 架构（Ampere、Hopper）声明支持范围；本仓库为部分算子提供 `*_maca.py` 专用实现，由 Op 层通过 `is_maca()` 分派。参见 [MetaX C500 上的算子可用范围](docs/summer-camp/README.zh-CN.md#12-metax-c500-上的算子可用范围)。

这种分层使面向用户的行为与 GPU 策略相互独立，AI Agent 和开发者可以修改其中一层，而不对另一层产生意外影响。

### 主要特性

- **规范驱动**——每个算子都由机器可读的 Manifest（`tileops/manifest/`）声明，
  其中定义签名、工作负载和 Roofline 公式，同时作为 AI Agent 生成代码与自动验证的入口。
- **Roofline 评估**——Kernel 性能依据硬件 Speed-of-Light 理论上限评估，而不是只与相对基线比较。
- **自动调优**——内置对 tile 大小、流水线和调度参数的搜索。
- **轻量依赖**——仅依赖 TileLang、PyTorch 和 einops。

## 安装

运行时需要支持 MXMACA 的 MetaX GPU。

### 前置条件

- Python >= 3.10
- PyTorch >= 2.1（MetaX 定制版，如 `2.8.0+metax3.7.1.3`）
- MetaX GPU：**C500**
- [TileLang](https://github.com/tile-ai/tilelang)：MetaX 环境使用容器内预编译的 MACA 版本

> [!WARNING]
> **MetaX 容器内不要执行 `make install`、`pip install tileops`，或任何会解析 tilelang 依赖的 pip 命令。**
>
> 容器里的 TileLang 是源码就地编译的 MACA 版本（例如 `/opt/tilelang-metax-v0.1.10`），
> 不是 pip 包（`pip show tilelang` 查不到）。因此 pip 会认为它「未安装」，从镜像源拉取
> 官方 **CUDA** 构建的 wheel 装进 site-packages，遮蔽 MACA 编译产物，导致 Kernel 编译走错后端。
>
> 同理，不要使用不带 `--system-site-packages` 的 `python3 -m venv`，否则会切断 MetaX 定制版
> PyTorch 和与 `libtilelang.so` ABI 耦合的 `apache-tvm-ffi`。
>
> 也不要在 pip 命令中带 `-c constraints.txt`：该文件的钉版面向 CUDA CI 环境，会降级
> `apache-tvm-ffi`，与容器内编译产物 ABI 不匹配。这类不匹配在 `import` 阶段看不出来，
> 要到第一次编译 Kernel 时才会失败。

### MetaX 环境：使用容器内预编译的 TileLang

不需要安装任何东西。设置 `PYTHONPATH` 后即可直接使用：

```bash
# 指向容器内预编译的 MACA 版 TileLang，以及本仓库根目录
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/path/to/TileOPs-Metax:$PYTHONPATH
```

`tileops` 无需 `pip install` 即可导入，Manifest 校验和测试都能直接运行。

如果确实需要把 `tileops` 注册进环境（例如想在仓库外的目录运行脚本），只能用 `--no-deps`：

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

`--no-deps` 是关键，它让 pip 不去解析 `tilelang` 依赖。仓库 CI 使用的 [`scripts/ci/install_tileops.sh`](scripts/ci/install_tileops.sh) 就是这个写法。

验证安装：

```bash
# 检查沐曦 GPU 状态。注意：若输出含 Sliced GPU 段落，实际可用显存和算力是切片配额，
# 不是第一段显示的整卡值
mx-smi
# 检查 Python 版本
python --version
# 检查 PyTorch 是否能识别 GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
# 检查 PyTorch 是 MetaX 定制版（版本号应含 metax）
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# 检查 TileLang 来自容器内的 MACA 构建，而不是 pip 装的官方 CUDA 版。
# 路径应指向 /opt/tilelang-metax-*；若指向 site-packages，说明已被覆盖，需要恢复
python -c "import tilelang; print(tilelang.__version__); print(tilelang.__file__)"
# 检查编译后端是 maca 而不是 cuda
python -c "from tilelang.utils.target import determine_target; print(determine_target('auto'))"
python -c "import einops; print('einops OK')"
```

## 快速开始

```python
import torch
from tileops.ops import GemmOp

M, N, K = 1024, 1024, 512

# GemmOp 是输入推断的：m/n/k 和 dtype 由 forward 的输入决定，构造时只声明布局。
# trans_b=False 表示 B 按 [K, N] 存储；默认值 True 对应 [N, K]。
gemm = GemmOp(trans_a=False, trans_b=False)

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C = gemm(A, B)          # [M, N]
```

> [!NOTE]
> 运行前需要先设置 `PYTHONPATH`（见上文安装小节）。
>
> 在 C500 上，`GemmOp` 通过 `is_maca()` 分派到 `tileops/kernels/gemm_maca.py`。
> 并非所有算子都有 MACA 实现，选择算子和工作负载前请先阅读
> [MetaX C500 上的算子可用范围](docs/summer-camp/README.zh-CN.md#12-metax-c500-上的算子可用范围)。

## 文档

设计文档和开发指南位于 [`docs/`](docs/) 目录。完整 API 参考和性能表发布在 [TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io)。

## 参与贡献

设计文档请参阅 [`docs/`](docs/)。分支与提交规范位于[`.claude/conventions/types.sh`](.claude/conventions/types.sh)。

## 许可证

TileOPs 使用 [MIT License](LICENSE) 发布。
