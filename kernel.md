# moe_expand_to_fused —— 待实现 Kernel 清单

对齐上游 `tile_kernels/moe/expand_to_fused_kernel.py`（`MetaX-MACA/TileKernels-Metax`，`dev`）。
上游用**一个**参数化 `prim_func` 覆盖全部路径，scale-factor 的块大小、打包与布局都是编译期特化，不是运行期分支。
本仓库按同样思路：**一个 Kernel 类 + 编译期特化**，不为每条路径写独立 kernel。

Manifest 同步为两个条目（同一 `source.kernel` / `source.op`，第二个是 `variant_of`）：

| Manifest 条目 | 激活 | scale factor | 状态 |
| --- | --- | --- | --- |
| `MoeExpandToFusedFwdOp` | fp16 / bf16 | 无 | 已有实现，待补 test/bench |
| `MoeExpandToFusedWithSFFwdOp` | fp8 (e4m3) / fp4 (e2m1) | fp32 或 UE8M0 打包 int32；row-major 或 TMA 对齐 col-major | 待实现 |

**不再按 scale-factor 打包方式拆条目**：上游把 `use_packed_ue8m0` 从 `x_sf.dtype == torch.int32` 推导出来，
不是独立参数，所以 fp32 与 packed-int32 合并为一个 `float32 | int32` dtype union。
代价是 `S` 无法按 dtype 分支约束，改用两种形式的**析取**规则来保住可机检性。

**FP4 不需要独立条目**：上游 `BaseCastConfig` 用 `torch.int8` 承载 e2m1，
`get_physical_hidden` 在入口把 hidden 减半，因此 `x.shape[1]` 在 fp8 / fp4 下都是**存储字节数**，
`elem_bytes` 都是 1，全部 shape 规则与 roofline 表达式逐字相同。

## K1. `MoeExpandToFusedKernel`（非量化路径）

文件 [tileops/kernels/moe/expand_to_fused.py](tileops/kernels/moe/expand_to_fused.py) — **已存在，178 行**。
特化参数 `(hidden, num_topk, dtype)`；`num_tokens` / `num_expanded_tokens` 为 `T.dynamic` 符号，batch 不触发重编译。

已覆盖：`pos_to_expert[p] < 0` 清零、`token_topk_to_pos[t, k] < 0` 跳过、
`num_blocks = max(num_tokens, num_expanded_tokens)`、`T.thread_return()` 提前退出、
`hidden` 向上对齐到 64 线程、`@functools.lru_cache(maxsize=32)`。

待补：

- [ ] `supported_archs` 目前是 `[80, 86, 89, 90]`，未含 MACA C500；确认 C500 的 arch 取值后补上。
- [ ] 非 64 对齐 `hidden` 的尾迭代确认：kernel 依赖越界写被 buffer extent 谓词化，需在 C500 上实测而非假定。

## K2. scale-factor 散射（`with_sf` 路径）

在 K1 的 prim_func 上增加编译期参数 `num_per_channels`、`use_tma_aligned_col_major_sf`、`use_packed_ue8m0`：

- [ ] 增加 `x_sf: T.Tensor((num_tokens, hidden_sf), sf_dtype)` 输入与 `expanded_x_sf` 输出。
- [ ] `hidden_sf = ceil_div(hidden, num_per_channels)`；`use_packed_ue8m0` 时再 `ceil_div(·, 4)`。同样向上对齐到线程数。
- [ ] 输出用 `T.StridedTensor`，步长含 `T.dynamic('sf_stride')` 运行期符号 —— col-major 时分配 `(hidden_sf, align(X, 4))`、切片到 `[:, :X]`、返回转置视图，逻辑 shape 与 row-major 一致但步长不同。
- [ ] 清零分支同步覆盖 sf 平面（`pos_to_expert[p] < 0` 时 sf 行也要清零）。
- [ ] 散射分支中 sf 与激活共用同一个 `pos_local[k]`，不重复读映射表。
- [ ] `num_per_channels ∈ {32, 128}`、`x_sf.dtype ∈ {float32, int32}`、以及 `int32 ⇒ col-major` 三条在 Op 层校验，不进 kernel。
- [ ] `use_packed_ue8m0` 由 `x_sf.dtype == torch.int32` 推导，不作为 Op 的显式参数暴露（与上游一致）。

打包对散射逻辑透明：kernel 搬运整个 int32 字，打包只改行长，不改每项宽度，因此 roofline 每项仍是 4 字节。

## K3. 激活 dtype 扩展（fp8 / fp4）

- [ ] `_SUPPORTED_DTYPES` 从 `(float16, bfloat16)` 扩到含 `float8_e4m3fn` 与 `int8`（承载 e2m1）。
- [ ] kernel 侧 dtype 透传即可 —— scatter 是 dtype 无关的行拷贝，body 中无一处依赖激活 dtype。
- [ ] fp4 的 `hidden` 传物理字节数（`H_logical / 2`），与上游 `get_physical_hidden` 一致；**不要**在本算子内做逻辑/物理换算，上游 `expand_to_fused_kernel.py` 也没有 import 那两个 helper。

## K4. Kernel 类与注册

- [ ] `__init__.py` 已导出 `MoeExpandToFusedKernel`（`__all__` + 显式 re-export，已完成）。
- [ ] 两个 manifest 条目共用 `kernel_map: {expand_to_fused_kernel: MoeExpandToFusedKernel}`；量化路径落地后确认是否仍为单一 Kernel 类，或需拆出 `MoeExpandToFusedSFKernel`（拆则两个条目的 `kernel_map` 要同步）。

## 现有 Op 层与 spec 的偏差（需修）

- [ ] [tileops/ops/moe/expand_to_fused.py:187](tileops/ops/moe/expand_to_fused.py#L187) 拒绝 `num_expanded_tokens > num_tokens * num_topk`。这比上游严：上游对二者不作任何断言，而 fused 布局按专家跨度补齐，小 batch 下 `X > T*K` 是常态（decode workload 就是 T=1、K=8、X=16）。该检查会误拒合法输入，需删除。
- [ ] 零填充语义与 torch 参考不一致：kernel 用 `torch.empty` 分配、只对 `pos_to_expert[p] < 0` 的行清零；torch 参考用 `torch.zeros` 全量清零。对于 `pos_to_expert[p] >= 0` 但未被任何 `token_topk_to_pos[t,k]` 指向的位置，两者结果不同（未初始化 vs 0）。写 test 时必须构造满足「每个已分配位置被恰好命中一次」的映射表，否则随机用例会 flaky。

## 不可在 manifest 中表达的约束（落到 test）

- [ ] `x_sf.dtype == int32 ⇒ use_tma_aligned_col_major_sf == True`（DSL 无法按 dtype 分支）。
- [ ] col-major 输出的真实 strides 与 `align(X, 4)` 补齐 —— `_VALID_LAYOUTS` 只有 `channels_last`，没有 col-major 成员，manifest 里只能声明逻辑 shape。
- [ ] Expert-major 分组是**前置条件**而非本算子的产物：位置轴的专家分组由上游 `get_fused_mapping` 建立，本算子只读 `pos_to_expert` 的符号位，专家 id 本身从不解引用。

## 下游（不属于 kernel，列出以免遗漏）

- [ ] `tests/ops/test_moe_expand_to_fused.py` —— 缺失，`source.test` 已指向。
- [ ] `benchmarks/ops/bench_moe_expand_to_fused.py` —— 缺失，`source.bench` 已指向。
- [ ] 两个条目均为 `status: spec-only`。翻 `implemented` 需同时满足 ≥2 workloads（已满足：8 / 10）与 test/bench 就位。
