<div align="center">
  <img src="assets/logo.png" width="300" alt="TileOPs-Metax Logo">

  <h1>TileOPs-Metax</h1>

  <p>
    <strong>面向大语言模型、由规范驱动的 GPU 算子库</strong>
  </p>

  <p>
    <strong>简体中文</strong> |
    <a href="README.en.md">English</a>
  </p>
</div>

> **项目状态**：TileOPs-Metax 正在积极开发中，API 可能发生变化。

## 首届开源英才夏令营

夏令营学员应使用 `summer-camp-2026` 分支，并遵循[算子迁移指南](docs/summer-camp/README.zh-CN.md)。

### 任务说明

学员以小组为单位，从筹备组发布的[候选算子清单](docs/summer-camp/TileKernels-MACA-待迁移算子盘点.md)中选择一个主算子。候选算子主要来源于 [`MetaX-MACA/TileKernels-Metax`](https://github.com/MetaX-MACA/TileKernels-Metax) 的 `dev` 分支，并已初步评估 MetaX C500 适配或性能优化空间。学员需要按照 TileOPs 的接口和工程规范完成算子泛化、迁移或优化，而不是简单复制源实现：

1. 认领算子并提交 Manifest；
2. 完成 Op、TileLang Kernel、正确性及边界测试；
3. 在真实 MetaX C500 上完成正确性验证、Benchmark、Profiler 和 Roofline 分析；
4. 提交可 Review 的实现 PR，并参加成果答辩。

完整时间安排、提交要求和验收标准见[算子迁移指南](docs/summer-camp/README.zh-CN.md)。

**可认领范围以筹备组发布的候选算子清单为准，源仓库中的算子不会默认全部开放认领。每组应优先选择一个中等或较高难度的主算子；在保证主算子完整交付并经助教确认后，最多可追加两个算子。每个算子须单独认领并分别提交 PR，不得提前占用尚未开始的算子。**

### 结营证书

按要求完成任务并参加成果答辩的学员，可获得本届开源英才夏令营结营证书。

### 优秀项目评选

答辩结束后，评委将结合小组答辩、PR、代码与测试质量、MetaX C500 验证证据、性能分析及开源复用价值，综合考虑算子难度、实际完成情况、优化效果和工程贡献，评出以下三个奖项：

- **最佳性能优化（1 组）**：性能分析准确，优化方案合理，在代表性工作负载上取得稳定且可信的性能提升。
- **最佳工程实践（1 组）**：算子交付完整，代码结构清晰，测试覆盖充分，PR 和复现材料规范。
- **最佳开源贡献（1 组）**：成果具有较高的复用、扩展或推广价值，能够帮助后续算子开发、硬件适配和社区协作。

原则上每组最多获得一个主奖。评奖不以绝对加速比作为唯一依据，评委将结合算子基础难度、工作量、完成质量和证据可信度综合判断。

## 项目简介

TileOPs-Metax 是一个基于 [TileLang](https://github.com/tile-ai/tilelang)、面向大语言模型训练和推理的 GPU 算子库。项目采用规范驱动的开发模式，帮助开发者和 AI Agent 构建、评估和优化高性能算子。

### 主要特性

- **规范驱动**：使用机器可读的 Manifest 定义算子签名和工作负载。
- **性能评估**：结合 Benchmark 与 Roofline 分析评估 Kernel 性能。
- **分层设计**：用户接口与硬件优化策略相互独立。

## 详细文档

- [查看完整中文说明](README.zh-CN.md)
- [View the full English README](README.en.md)
- [查看夏令营算子迁移指南](docs/summer-camp/README.zh-CN.md)
