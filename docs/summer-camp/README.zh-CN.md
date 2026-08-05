# 首届开源英才夏令营算子迁移指南

[**简体中文**](README.zh-CN.md) | [English](README.en.md)

本专项面向 2026 年 8 月 3 日至 8 月 6 日线下夏令营。学员从筹备组发布的候选算子清单中选择课题，对具有通用价值的 TileLang Kernel 进行泛化、MetaX C500 适配、测试验证和性能优化，并按照 TileOPs 的 Manifest → Test → Op/Kernel → Benchmark 信任链提交可复现、可维护的开源成果。

> **C500 验收基线**：代码编辑、文档编写、Manifest 校验和格式检查可以在其他环境完成；最终 Kernel 编译与运行、正确性/边界/异常测试、Benchmark、mcProfiler、Roofline 实测及 PR 验收证据必须来自真实沐曦 MetaX C500。

## 时间安排与完成标准

| 时间 | 里程碑 |
|---|---|
| 8 月 3 日 | 完成环境验证、阅读规范，并从候选清单认领一个主算子 |
| 8 月 4 日 | 提交并通过 Manifest PR 的快速 Review；创建实现 PR，并通过基础正确性测试 |
| 8 月 5 日 18:00 前 | 实现 PR 达到可 Review 状态，测试和 C500 性能验证证据完整 |
| 8 月 5 日晚 | 助教完成初步检查，并在 PR 中列出需要修复的阻塞问题 |
| 8 月 6 日 10:30 前 | 完成阻塞问题修复；11:00 冻结参评版本并确定答辩名单 |
| 8 月 6 日下午 | 进行成果答辩，展示算子实现、正确性、性能优化和开源价值 |

任务完成须同时满足：Manifest 已合入且可校验；Op 与 Kernel 分层清晰；正确性、边界和异常测试通过；独立基线 Benchmark 可运行；Roofline 公式和实测结果可解释；PR 模板及 C500 证据完整；所有 Review 阻塞项已经关闭。

## 1. 准备环境

需要 Python 3.10+、Git、可用的 MetaX 驱动/运行时和 MetaX GPU。推荐在筹备组提供的容器中工作。

> [!TIP]
> **推荐使用夏令营专属在线环境**：通过[模力方舟算力市场](https://ai.gitee.com/compute)租用 MetaX C500 算力，并选择夏令营专属镜像：`PyTorch-Agent / 2.8.0 / Python 3.12 / MACA 3.7.1.5`。镜像库存、价格及页面名称以平台实际显示为准。创建实例后，仍须按照第 1.1 节完成环境自检，并通过 `mx-smi` 确认实际设备为 MetaX C500。

> [!WARNING]
> **不要执行 `make install`、`pip install tileops`、`pip install -e '.[dev]'`（不带 `--no-deps`），
> 也不要创建不带 `--system-site-packages` 的 venv。**
>
> 容器里的 TileLang 是源码就地编译的 MACA 版本（例如 `/opt/tilelang-metax-v0.1.10`），
> 不是 pip 包，`pip show tilelang` 查不到。因此 pip 会认为它「未安装」，从镜像源拉取官方
> **CUDA** 构建的 wheel 覆盖它；新建的 venv 则会切断 MetaX 定制版 PyTorch。
> 两种情况都需要重新编译或重装才能恢复，属于高危操作。
>
> 同理不要在 pip 命令里带 `-c constraints.txt`：该文件的钉版面向 CUDA CI 环境，
> 会降级与 `libtilelang.so` ABI 耦合的 `apache-tvm-ffi`。这类不匹配在 `import` 阶段看不出来，
> 要到第一次编译 Kernel 时才失败。

### 1.1 环境准备与自检

`tileops` 不需要安装，设置 `PYTHONPATH` 后即可直接导入并运行测试。

开始前，请先在 GitLink 上 Fork 官方仓库到个人账号，再克隆个人 Fork。以下命令中，`origin` 为个人 Fork，`upstream` 为官方仓库：

```bash
git clone https://www.gitlink.org.cn/<your-account>/TileOPs-Metax.git
cd TileOPs-Metax
git remote add upstream https://www.gitlink.org.cn/ccf-ai-infra/TileOPs-Metax.git
git fetch upstream
git switch -c summer-camp-2026 --track upstream/summer-camp-2026
git pull --ff-only

# 指向容器内预编译的 MACA 版 TileLang（请按容器实际路径调整），以及本仓库根目录
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:$PYTHONPATH

# 自检：TileLang 必须来自 /opt/tilelang-metax-*，后端必须是 maca。
# 若路径指向 site-packages 或后端是 cuda，说明环境已被 pip 覆盖，需要先恢复
python -c "import tilelang; print(tilelang.__version__); print(tilelang.__file__)"
python -c "from tilelang.utils.target import determine_target; print(determine_target('auto'))"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mx-smi

# 验证仓库可用
python scripts/validate_manifest.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
```

如果确实需要在仓库外的目录运行脚本，只能用 `--no-deps` 安装（`--no-deps` 让 pip 不去解析 tilelang）：

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

如果基础环境自检失败，先记录操作系统、Python、驱动、MACA、GPU 型号、失败命令和退出码。不要在同一个 PR 中混入环境修复和算子迁移。

### 1.2 MetaX C500 上的算子可用范围

选题**之前**必须读这一节，否则可能认领一个在 C500 上根本跑不起来的算子。

**MACA 专用 Kernel 与分派机制。** 本仓库为部分算子提供了 MACA 专用实现，通过 `tileops/utils/utils.py` 的 `is_maca()` 在 Op 层分派：

```python
# tileops/ops/attention/deepseek_dsa.py
if is_maca():
    kernel_cls = SparseMlaMACAKernel
elif is_hopper():
    kernel_cls = SparseMlaKernel
```

现有的 MACA 专用 Kernel：

```text
tileops/kernels/gemm_maca.py
tileops/kernels/grouped_gemm/grouped_gemm_persistent_maca.py
tileops/kernels/moe/moe_grouped_gemm_persistent_fused_act_maca.py
tileops/kernels/moe/shared_expert_mlp_maca.py
tileops/kernels/reduction/argreduce_maca.py
tileops/kernels/deltanet/compute_w_u_bwd_maca.py
tileops/kernels/deltanet/deltanet_bwd_maca.py
tileops/kernels/gated_deltanet/gated_deltanet_prefill_maca.py
```

**arch 门禁。** C500 上 `torch.cuda.get_device_capability()` 上报 `(8, 0)`，因此`get_sm_version()` 返回 `80`。这个数字沿用 NVIDIA SM 编号语义，**与 C500 的实际架构无关**，仅用于 Kernel 门禁比对；不要据此推断「C500 相当于 Ampere」。

`supported_archs` 不含 `80` 的 Kernel 声明共 20 处（`[90]` 17 处、`[89, 90]` 3 处），分布在下列文件：

```text
attention/deepseek_dsa_decode.py    attention/deepseek_mla_decode.py
attention/gqa_bwd.py                attention/gqa_decode_bs1.py
attention/gqa_fwd.py                attention/gqa_fwd_fp8.py
attention/gqa_fwd_ws.py             attention/gqa_prefill_fwd_ws.py
attention/gqa_sliding_window_fwd.py attention/gqa_sliding_window_varlen_fwd.py
bmm.py                              gemm.py
deltanet_recurrence.py              gated_deltanet_recurrence.py
grouped_gemm/grouped_gemm_persistent.py
grouped_gemm/grouped_gemm_persistent_3wg.py
moe/moe_grouped_gemm_persistent_3wg_fused_act.py
```

它们依赖 Hopper 专属特性，例如 warp-specialization barrier intrinsic `ptx_init_barrier_thread_count`；强行绕过门禁会在 lowering 阶段失败：

```text
tvm.error.InternalError: Unresolved call ir.Op(name="tirx.ptx_init_barrier_thread_count", ...)
```

> [!IMPORTANT]
> **Kernel 被门禁挡住不等于 Op 不可用。** `gemm.py` 的 `GemmKernel` 声明 `[89, 90]`，
> 在 C500 上确实被挡；但 `GemmOp` 通过 `is_maca()` 分派到 `gemm_maca.py`
> （`supported_archs = [80, 86, 89, 90]`），因此 **`GemmOp` 在 C500 上可用**，
> 实测 `M,N,K` 取 1024³ 和 4096³ 均通过。
>
> 判断一个算子能否在 C500 上使用，**要看 Op 层实际分派到哪个 Kernel，而不是只看某个
> Kernel 的 `supported_archs`**。最可靠的方式是直接构造 Op 并运行。

若某个 Op 在 C500 上仍抛下列异常，说明它没有 MACA 分派路径，不适合作为迁移选题：

```text
ValueError: BmmFp8Kernel is not supported on architecture 80
```

注意这条信息只给出 `80` 这个数字，不含设备名，容易让人误以为身处 NVIDIA Ampere 卡。在 C500 上看到 `architecture 80` 时，请按上文理解：它只是 `get_sm_version()` 的返回值。

查询方式：

```bash
grep -rn "supported_archs" tileops/kernels/       # 门禁声明
grep -rn "is_maca" tileops/ops/                   # 已有 MACA 分派的 Op
ls tileops/kernels/**/*maca*.py                   # 已有的 MACA 专用 Kernel
```

为尚无 MACA 实现的算子补一个 `*_maca.py` Kernel 加 `is_maca()` 分派，是合适的迁移选题方向。

**Op 可用不等于任意 shape 可用。** 归约类算子在 C500 上还有实测的形状上限。以 `SoftmaxFwdOp`（`supported_archs` 含 80，无需 MACA 分派）为例：

| 输入 shape | 结果 |
|---|---|
| `(128, 128)` / `(512, 512)` / `(1024, 1024)` | OK |
| `(4096, 1024)` / `(8192, 1024)` | OK |
| `(1024, 1536)` | 失败：`no available layout`（layout 推断失败） |
| `(1024, 2048)` / `(2048, 2048)` / `(4096, 4096)` | 失败：`MACALaunch Error: mcErrorInvalidValue` |

瓶颈在**归约维度**而非行数：行数 8192 可用，归约维度超过 1024 即失败。编写测试矩阵和 Benchmark workload 时，请先用小尺寸确认可用范围再放大，并把实测到的形状上限写进 PR 证据。

### 1.3 已知环境问题

**父子进程重复导入 TileLang 会被 SIGKILL。** 在已 `import tilelang` 的进程中再用 `subprocess` 启动一个也会导入 tilelang 的子进程，整个进程组会被 SIGKILL（`exit 137`，**没有任何 traceback 或错误输出**）。

因此下列命令在 C500 上会以 `exit 137` 中断，这不是你的代码问题：

```bash
python -m pytest -q tests/test_validate_manifest.py     # 卡在约 59% 后 exit 137
```

校验器本身是正常的，直接运行即可，或跳过该用例：

```bash
python scripts/validate_manifest.py     # exit 0
python -m pytest -q tests/test_validate_manifest.py --deselect \
  "tests/test_validate_manifest.py::TestIntegration::test_validator_passes_on_current_codebase"
```

最小复现（供上游排查参考）：

```bash
# 父进程导入 tilelang，子进程也导入 -> SIGKILL
python -c "
import tilelang, subprocess, sys
r = subprocess.run([sys.executable,'-c','import tilelang'], capture_output=True, text=True)
print('rc =', r.returncode)
"
# 父进程不导入，或子进程不导入，均正常
```

`benchmarks/benchmark_base.py` 和 `benchmarks/hardware/memory/hbm_bandwidth.py` 也使用 subprocess，做 Benchmark 时如遇无输出的 `exit 137`，优先怀疑这个问题。

**arch 门禁产生 failed 而非 skipped。** 被门禁拦住的算子，其纯参数校验测试（如 `test_bmm_fp8_batch_mismatch_raises`）也会报 `failed` 而不是 `skipped`，因为 `ValueError` 在 Op 构造阶段就抛出，测试没走到断言。例如 `pytest -q -m smoke tests/ops/test_bmm.py` 在 C500 上实测为 `13 failed, 8 passed`。提交证据时请注明哪些失败源于环境门禁、哪些源于自己的实现。

## 2. 认领算子

1. 可认领算子以筹备组发布的[候选算子清单](TileKernels-MACA-待迁移算子盘点.md)为准。候选算子主要来源于 [`MetaX-MACA/TileKernels-Metax`](https://github.com/MetaX-MACA/TileKernels-Metax) 默认 `dev` 分支，但源仓库中的算子不会默认全部开放认领。
2. 每组应优先选择一个中等或较高难度的算子作为主算子。候选清单中的不同算子均可独立认领。
3. 每组参照 [算子认领说明 Issue #1](https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax/issues/1) 创建一个独立的算子认领 Issue，填写小组编号、算子名称、源文件路径和源提交 SHA。认领信息完整、该算子未被其他小组认领并经助教确认后，认领方才有效；同一算子出现多个认领 Issue 时，以最先提交完整信息并经助教确认的 Issue 为准。
4. 发现源实现不完整、依赖缺失或迁移范围过大时，立即在 Issue 中说明；不得静默换题。

**主算子的 Manifest PR 已合入、实现 PR 已通过核心正确性测试且不存在阻塞问题，并经助教确认后，每组最多可追加两个算子，即累计最多认领三个算子。每个算子须单独认领并分别提交 PR，不得提前占用尚未开始的算子。**

## 3. 使用两个 PR 建立信任链

本节说明两个 PR 的职责和先后关系：PR A 先确定规范，PR B 再完成实现、测试和性能验收。PR A 和 PR B 使用同一个 `feat/<operator-id>` 分支，并均须链接本组的算子认领 Issue。具体提交格式和检查命令见第 7 节。

从最新的 `summer-camp-2026` 创建开发分支：

```bash
git switch summer-camp-2026
git pull --ff-only
git switch -c feat/<operator-id>
```

### PR A：Manifest

PR A 用于在实现前确定算子的接口、数据类型、形状规则、工作负载和 Roofline 公式，作为后续实现、测试与 Benchmark 的统一契约。

第一阶段只提交：

- 在 `tileops/manifest/<family>.yaml` 中新增对应算子；仅在没有合适 family 时创建新的 Manifest 文件；
- Manifest 校验或必要的契约测试；
- 对输入输出、shape、dtype、工作负载和 Roofline 公式的说明。

新 Manifest 的初始状态必须是 `spec-only`。PR A 必须经过助教或维护者的快速 Review，并在 Manifest 校验通过后合入。PR A 不审核 Op、Kernel、性能或 C500 数据。

PR A 合入前，不得在该分支提交实现代码。

### PR B：实现

PR A 合入后，继续使用原来的 `feat/<operator-id>` 分支。由于合入后的 PR A 可能生成新的提交 SHA，需要跳过本地 PR A 提交，将后续实现建立在最新目标分支上：

```bash
git switch feat/<operator-id>
git fetch upstream
git rebase --onto upstream/summer-camp-2026 <local-pr-a-sha> feat/<operator-id>
git push --force-with-lease origin feat/<operator-id>
```

`<local-pr-a-sha>` 是该分支中提交 PR A 时的 Commit SHA。变基后，目标分支之外应只保留 PR B 的实现提交。

同步完成后提交：

- `tileops/ops/` 下的无状态 Op；
- `tileops/kernels/` 下的 TileLang Kernel；
- `tests/` 下的正确性、边界和异常测试；
- `benchmarks/ops/` 下的独立基线 Benchmark；
- Manifest 中允许随实现更新的状态、来源和工作负载字段。

完成实现、测试和 C500 性能验证后，再从同一分支创建 PR B。

不要把无关重构、依赖升级或多个算子放进同一个 PR。

## 4. 迁移要求

- 先固定参考语义和失败行为，再编写实现。
- Op 负责参数校验、dtype/layout 处理和 Kernel 调度；Kernel 只负责设备计算，不承担用户接口职责。
- 不直接复制源仓库的测试结论；通过本仓库的测试入口重新建立验证证据。
- 跨文件修改须保持一个最小闭环：一个 Manifest、一个 Op、一个或少量策略 Kernel、一组测试和一个 Benchmark。
- 测试应先因目标行为尚未实现而失败，再补充实现；路径或语法错误不属于有效失败。
- 不得使用 PyTorch 或其他高层框架在 Host 侧代替应由 TileLang Kernel 完成的设备计算。

## 5. 正确性与测试

最低测试矩阵包括：

- 代表性常规形状；
- 非整块、最小值等边界形状；
- 所有声明支持的 dtype；
- 非连续输入（接口声明支持时）；
- 非法维度、dtype、shape 的明确异常；
- 与独立 PyTorch 参考实现比较，并注明 `atol`/`rtol`；
- 在真实沐曦 MetaX C500 上完成最终 GPU 测试。

常用命令（运行前确认已按第 1.1 节设置 `PYTHONPATH`）：

```bash
python scripts/validate_manifest.py
python -m pytest -q tests/<test_file>.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
pre-commit run --all-files
```

`tests/test_validate_manifest.py` 在 C500 上会因已知环境问题 `exit 137`，处理方式见第 1.3 节。

在 PR 中记录被测提交 SHA、完整命令、退出码和简洁结果，不提交巨大的原始日志。

## 6. Benchmark、mcProfiler 与 Roofline

Benchmark 必须与正确性测试分开，并使用独立基线，通常为 PyTorch 原语或清晰的参考组合。Benchmark、mcProfiler 和 Roofline 实测必须在真实沐曦 MetaX C500 上完成，并至少记录：

- 预热次数、测量次数、同步方式和统计方法；
- 输入 shape、dtype、布局和设备；
- TileOPs 延迟、基线延迟和加速比；
- mcProfiler 观测到的主要瓶颈，以及优化措施与瓶颈的对应关系；
- Manifest Roofline 公式所需的 FLOPs 和读写字节数；
- `achieved / theoretical` 比值及瓶颈判断；
- 被测提交 SHA、完整命令、软件版本、驱动版本和 GPU 信息；
- sGPU 切片配额（见下）。

> [!IMPORTANT]
> **注意 sGPU 切片。** 容器可能分到 GPU 切片而非整卡。执行 `mx-smi` 时不要只看第一段的整卡
> 显存（如 65536 MiB），要看 Sliced GPU 段落的 `Vram Quota` 和 `Compute` 百分比——例如
> 16000 MiB 配额 + 25% 算力。此时 `torch.cuda.get_device_properties(0).total_memory`
> 也只报切片值。
>
> 提交 Roofline 证据时必须记录切片配额，并说明 `P_peak` / `BW_peak` 取的是整卡值还是按切片
> 比例折算的值。否则用切片实测值除以整卡理论峰值，会得到偏低到无法解释的
> `achieved / theoretical`。Manifest 中的大 workload 在切片显存下也可能 OOM。

不要用被测实现充当基线，不要只报告最快一次，也不要把编译时间混入稳定态延迟。

## 7. 提交 PR

### 课题命名

课题名称用于答辩和评奖，PR 标题用于标识具体算子。

- 单算子课题建议命名为：`面向 MetaX C500 的 <算子名> 迁移与优化`。
- 多算子课题可按算子类别或共同功能命名，并注明主算子和追加算子。
- 每个 PR 的标题仍须使用对应算子名称，保持“一算子一 PR”。

示例：

```text
课题名称：面向 MetaX C500 的 MoE 路由多算子迁移与优化
主算子：moe_group_count
追加算子：moe_normalize_weight、moe_reduce_fused

PR 标题：
[moe_group_count] feat: 新增路由分组计数算子
[moe_normalize_weight] feat: 新增权重归一化算子
```

### PR A：Manifest PR

提交 PR A 时，必须使用仓库统一的 [Manifest PR 模板](../../.github/PULL_REQUEST_TEMPLATE/operator-manifest.zh-CN.md)。

PR 标题建议：

```text
[算子名] feat: 新增 spec-only Manifest
```

PR A 须说明算子名称、源文件路径、源提交 SHA、接口定义、工作负载和 Roofline 公式，不要求填写精度、性能、mcProfiler 或 C500 验证数据。

提交前执行：

```bash
git diff --check
python scripts/validate_manifest.py
```

### PR B：实现 PR

提交 PR B 时，必须完整使用仓库统一的[算子迁移 PR 模板](../../.github/PULL_REQUEST_TEMPLATE/operator-migration.zh-CN.md)，不得删除模板中的必填栏目。

PR 标题建议：

```text
[算子名] feat: 本次新增功能简短说明
[算子名] optimize: 本次优化简短说明
```

新增算子使用 `feat`；优化仓库中已有实现使用 `optimize`。

提交前执行：

```bash
git diff --check
python scripts/validate_manifest.py
python -m pytest -q <本算子测试>
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

PR B 必须完整填写小组课题信息、优化方案、精度验证、优化前后性能、加速比和 mcProfiler 瓶颈分析，并保留算子名称、源文件路径、源提交 SHA、测试提交 SHA、完整测试命令及 MetaX C500 验证证据。

### 提交前自查

在将 PR B 转为可 Review 状态前，各小组须在 [PR 提交前检查 Issue #4](https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax/issues/4) 中按模板完成自查；未完成项应如实填写，不得提前勾选。

## 8. Review 与汇报

PR A 进行快速 Review，重点检查 Manifest 接口、shape/dtype、工作负载、Roofline 公式和校验结果。

PR B 进行完整 Review：

- 技术 Review 检查参考语义、Op/Kernel 分层、测试矩阵、Benchmark 公平性、mcProfiler 分析和 Roofline 解释；
- 流程 Review 检查认领状态、PR 范围、模板完整度、C500 证据可复现性和阻塞项。

最终汇报建议用五分钟说明：

1. 算子解决什么问题；
2. 从哪里迁移、改了什么；
3. 如何证明正确；
4. 在 MetaX C500 上表现如何、离 Roofline 多远；
5. 下一步最值得优化什么，以及成果如何被其他贡献者复用。

遇到阻塞时，在认领 Issue 中给出“命令 + 退出码 + 最小日志 + 已尝试方法”，并 @当值维护者。切勿提交密码、Token、私钥、容器地址或完整环境变量。
