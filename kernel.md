# moe_expand_to_fused 算子迁移与 C500 优化文档

本文面向 `MoeExpandToFusedFwdOp` / `MoeExpandToFusedWithSFFwdOp` 的迁移、维护和后续优化，覆盖：

- Kernel 实现：`tileops/kernels/moe/expand_to_fused.py`
- Op 封装：`tileops/ops/moe/expand_to_fused.py`
- C500 架构参考：`.agents/skills/c500-guide/ch2.曦云C500芯片架构.md`

当前版本是一个 token-centric scatter/copy 算子：每个 token row 被复制到 `token_topk_to_pos[t, k]` 指定的 fused expert-major 位置；算子不排序、不做 GEMM、不解释 activation dtype。

## 1. 算子语义

### 1.1 输入输出

非 SF 路径：

| 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `x` | `[T, H]` | `float16` / `bfloat16` | token activation |
| `token_topk_to_pos` | `[T, K]` | `int32` | `(token, topk_slot) -> expanded position` |
| `pos_to_expert` | `[X]` | `int32` | expanded position -> expert id；负数表示 unassigned row |
| `expanded_x` | `[X, H]` | same as `x` | fused expert-major activation buffer |

SF 路径额外搬运 `x_sf`：

| 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `x` | `[T, H]` | `float8_e4m3fn` / `uint8` / `int8` | quantized activation；fp4 用 byte-packed 物理 H |
| `x_sf` | `[T, S]` | `float32` / `int32` | scale factor；`int32` 表示 packed UE8M0 |
| `expanded_x` | `[X, H]` | same as `x` | expanded activation |
| `expanded_x_sf` | `[X, S]` logical view | same as `x_sf` | expanded scale factor |

`S = ceil(H / num_per_channels)`；当 `x_sf.dtype == int32` 时，四个 UE8M0 exponent byte 打包到一个 `int32`，所以 `S = ceil(ceil(H / num_per_channels) / 4)`。

### 1.2 sentinel 和 zero-fill 约定

- `token_topk_to_pos[t, k] < 0`：该 token-expert slot 被 dropped，kernel 跳过。
- `pos_to_expert[p] < 0`：该 expanded row 是 unassigned/padding，kernel 对 `expanded_x[p, :]` 清零；SF 路径也同步清零 `expanded_x_sf[p, :]`。
- `pos_to_expert[p] >= 0` 但没有任何 `(t, k)` 指向 `p` 时，当前 kernel 不会清零该 row；调用方/路由器应保证已分配 position 被恰好写入，测试用例也应覆盖这个前置条件。
- `X > T * K` 是合法场景，例如 decode workload 需要 padded capacity；`num_blocks = max(T, X)` 正是为了同时覆盖 token scatter 和 tail zero-fill。

### 1.3 layout 约定

- activation output 永远是 row-major logical shape `[X, H]`。
- row-major SF：`expanded_x_sf[p, s]` 写入 `[X, S]`。
- TMA-aligned col-major SF：底层分配 `[S, align(X, 4)]`，kernel 通过 `T.StridedTensor` 写 `expanded_x_sf[s, p]`，Op 层返回 `.T` 后的 logical `[X, S]` view；packed UE8M0 必须走这个布局。

## 2. Kernel 层迁移说明

### 2.1 当前 Kernel 结构

`_expand_to_fused_kernel(...)` 是一个 compile-time specialized TileLang JIT builder，静态参数包括：

| 参数 | 是否静态 | 用途 |
| --- | --- | --- |
| `hidden` | 是 | activation row width |
| `num_topk` | 是 | 每个 token 的 expert slot 数 |
| `num_per_channels` | 是/可空 | 是否启用 SF path 以及 SF block width |
| `use_tma_aligned_col_major_sf` | 是/可空 | SF 输出是否为 TMA col-major |
| `use_packed_ue8m0` | 是/可空 | `int32` SF packed path |
| `x_dtype` / `sf_dtype` | 是 | TileLang buffer dtype |
| `num_tokens` / `num_expanded_tokens` | 否，`T.dynamic` | batch size 和 expanded capacity，不触发重编译 |
| `sf_stride` | 否，`T.dynamic` | TMA col-major view 的 runtime leading stride |

实现要点：

- `_NUM_THREADS = 64`，`hidden` 和 `hidden_sf` 都向上对齐到 64，保证每个 thread 的 loop trip count 规整。
- `with T.Kernel(num_blocks, threads=_NUM_THREADS)`：一维 grid，block id 同时表示 token id 或 expanded position id。
- `pos_local = T.alloc_local((num_topk,), "int32")`：每个 block 先把该 token 的 K 个输出位置加载到 local。
- 对 `pid_token < X && pos_to_expert[pid_token] < 0` 的 block 执行 zero-fill；这里保持嵌套 `if`，避免 TileLang 将 `and` 降成非短路谓词导致越界读 `pos_to_expert`。
- 对 `pid_token >= T` 的 block 用 `T.thread_return()` 退出；这些 block 只负责 padding row 清零。
- 对有效 token：先 `T.copy(x[pid_token, :], x_fragment[0:hidden])`，再串行遍历 `K`，把同一 row scatter 到多个 `pos_local[k]`。
- SF path 与 activation 共享同一个 `pos_local[k]`，避免重复读 routing map。

### 2.2 迁移到新 backend/新算子时必须保留的 contract

迁移时不要只看普通 case，应逐项保留以下行为：

- batch 维动态：`T` / `X` 不应进入 kernel cache key；只让 `H`、`K`、dtype 和 SF layout 触发重编译。
- dropped slot：`token_topk_to_pos < 0` 必须跳过，不写任意 output row。
- unassigned row：`pos_to_expert < 0` 必须清零 activation 和 SF。
- padded SF token axis：TMA col-major 底层 token dimension 必须 `align(X, 4)`，但返回给 Python 的 logical view 是 `[X, S]`。
- packed UE8M0：`x_sf.dtype == int32` 推导 packed path，且必须要求 `use_tma_aligned_col_major_sf=True`。
- fp4 physical hidden：fp4 由 `uint8` / `int8` byte storage 表示，kernel 收到的 `H` 是物理字节数，不在本算子内做 logical-H 到 physical-H 换算。
- zero-token batch：Kernel 类里 `x.shape[0] > 0` 时才 launch；Op 层目前要求 `T > 0`，如果后续要支持空 batch，需要先统一 Python validation 和 kernel behavior。

### 2.3 Kernel 类状态

| 类 | 状态 | 说明 |
| --- | --- | --- |
| `MoeExpandToFusedKernel` | 已实现 | 非量化 bf16/fp16 scatter；内部调用同一个 JIT builder，但 `num_per_channels=None`，SF 参数传 `None`。 |
| `MoeExpandToFusedWithSFKernel` | 已实现 | fp8/fp4 byte payload + fp32/int32 SF；支持 row-major SF 和 TMA col-major SF。 |

当前 `supported_archs` 仍是 `[80, 86, 89, 90]`，如果 TileOps manifest/runner 需要显式 C500 arch id，应在确认 C500 arch 编号后补充，而不是在文档里猜。

## 3. Op 层迁移说明

### 3.1 Op 封装职责

`tileops/ops/moe/expand_to_fused.py` 不写 kernel body，它负责：

- CUDA residency、device 一致性、rank/shape 和 dtype 校验。
- 将输入转为 contiguous，保证 kernel 的 row-major tensor contract。
- 维护 `_kernel_cache`，避免重复 JIT 编译。
- 绑定 `x_shape`、`token_topk_to_pos_shape`、`pos_to_expert_shape` 等 roofline 元数据。
- 根据 runtime dtype/shape 获取对应 Kernel 类实例。

### 3.2 cache key

非 SF：

```python
key = (hidden, num_topk, dtype)
```

SF：

```python
key = (hidden, num_topk, dtype, sf_dtype)
```

注意 `num_tokens`、`num_expanded_tokens` 和 `sf_stride` 都是动态符号，不应该进入 cache key。`num_per_channels` 和 `use_tma_aligned_col_major_sf` 是 Op 构造参数，已经固定在 op instance 上。

### 3.3 dtype 与 shape 规则

非 SF：

- `x.dtype in {torch.float16, torch.bfloat16}`。
- `token_topk_to_pos.dtype == torch.int32`。
- `pos_to_expert.dtype == torch.int32`。

SF：

- `x.dtype in {torch.float8_e4m3fn, torch.uint8, torch.int8}`。
- `x_sf.dtype in {torch.float32, torch.int32}`。
- `x_sf.dtype == torch.int32` 时必须 `use_tma_aligned_col_major_sf=True`。
- `x_sf.shape[1]` 必须等于 `_expected_hidden_sf(hidden, x_sf.dtype)`。

### 3.4 roofline 字节模型

非 SF：

```text
bytes = T * H * elem_bytes
      + T * K * 4
      + X * 4
      + X * H * elem_bytes
flops = 0
```

SF：

```text
bytes = T * H * elem_bytes
      + T * S * 4
      + T * K * 4
      + X * 4
      + X * H * elem_bytes
      + X * S * 4
flops = 0
```

这是 memory-only scatter/copy 算子，不能用 TFLOPS 判断优劣；主要看 latency、有效带宽、launch 占比和下游 pipeline 是否重复读写同一 materialized buffer。

## 4. C500 软硬协同分析

### 4.1 C500 相关硬件事实

来自 C500 架构章节的关键参数：

| 项目 | C500 参数 | 对本算子的影响 |
| --- | ---: | --- |
| warp/wave size | 64 lanes | 当前 `_NUM_THREADS=64` 正好是一整个 wave；控制简单，但单 block 只用一个 wave 搬一整行。 |
| AP/SM 数量 | 104 | 小 batch `num_blocks` 少时很难填满设备。 |
| vector register file | `64 lanes * 2048 * 4B = 512KB/AP` | 单 row staging 放在 fragment/register 中是可行的；profile 中 register 改善不等于端到端收益。 |
| scalar register file | `800 * 4B` | routing index 和控制流占用标量资源，但不是主流量。 |
| shared memory / WSM | 64KB/AP，约 128B/cycle/AP，约 60 cycles | H=7168 bf16 单行约 14KB，可放入 WSM；但当前 token row 只在一个 block 内复用 K 次，fragment/register staging 已经满足，不一定值得再走 shared。 |
| Vector L1 | 32KB/AP，约 60 cycles，默认关闭 | 不能假设 VL1 会缓存 expanded output；优化应围绕 coalescing 和减少 DRAM 往返。 |
| constant L1 | 8KB/AP | `K` 很小，routing/常量可受益，但总字节占比低。 |
| L2 | 8MB，全 AP 共享，约 170~200ns | 大 shape 的 `expanded_x` 远超 L2，下游 GEMM 很难从 L2 复用 standalone scatter 的输出。 |
| DRAM | 64GB HBM，延迟 400ns+，参考带宽 1843.2 | materialized scatter 是主要成本；DRAM-L2 是半双工，读写往返会互相争用。 |

C500 的 vector load 聚合规则是每 16 个 lane 聚合到 128B cacheline transaction。对本算子来说，读 `x[t, :]` 和写 `expanded_x[p, :]` 都是 row-contiguous，有空间局部性；但如果每 lane 只搬很窄的元素，transaction 利用率可能不足，需要通过向量化/更宽连续搬运来验证。

### 4.2 数据规模与 cache 不匹配

典型 output footprint：

| H | X | dtype | `expanded_x` | vs 8MB L2 |
| ---: | ---: | --- | ---: | ---: |
| 7168 | 4096 | bf16/fp16 | 56MiB | 7.0x |
| 7168 | 32768 | bf16/fp16 | 448MiB | 56.0x |
| 3072 | 4096 | bf16/fp16 | 24MiB | 3.0x |
| 3072 | 32768 | bf16/fp16 | 192MiB | 24.0x |

结论：standalone `expand_to_fused` 写出的 `[X, H]` buffer 对 L2 来说太大。下游 grouped GEMM 再读 `expanded_x` 时，大概率是新的全局内存流量，而不是便宜的 L2 hit。这个事实比单 kernel thread tuning 更重要。

### 4.3 Little's Law 视角

C500 指南强调带宽、延迟和并发请求数之间的关系。`expand_to_fused` 的问题可以拆成两类：

- 小 batch：`num_blocks` 少，AP occupancy 不足，launch/runtime overhead 与 kernel body 同量级；增加单 block 搬运并行度只能改善 body，不能消除 launch。
- 大 batch：有足够 blocks，但每个 token 需要把同一 row 写到 K 个输出位置，DRAM 写流量和下游重读流量主导；并发能填满带宽后，继续增加 threads 可能只是在同一个半双工 DRAM-L2 通路上排队。

因此，优化目标应从“提高单 kernel 内并发”转向“减少必须经过 DRAM 的字节数”。

## 5. 当前瓶颈

### 5.1 token-centric scatter 的成本

当前 kernel 的核心路径是：

```text
load token_topk_to_pos[t, :]
load x[t, :] into fragment
for k in 0..K-1:
    if pos[t,k] >= 0:
        write expanded_x[pos[t,k], :]
        optionally write expanded_x_sf[pos[t,k], :]
```

优点：

- 一次读取 `x[t, :]`，复用到 K 个 routed expert row。
- 支持任意 dropped slot 和 padded capacity。
- 对 routing 语义要求低，容易和现有上游 map 对齐。

代价：

- `K` 是串行 loop；K=8 时每个 token row 至少执行 8 次 output row store。
- 写出的 `[X, H]` 只是下游 GEMM 的中间态，会被重新读一遍。
- 对 dense sequential routing，通用 `token_topk_to_pos` 读取和 top-k serial scatter 是额外开销。
- 小 batch 下 block 数不足，C500 104 个 AP 无法被充分填满。

### 5.2 SF path 的成本

SF 字节量通常远小于 activation，例如 H=7168、`num_per_channels=128` 时 `S=56`，`X=4096` 的 fp32 SF 约 0.875MiB，而 activation output 是 56MiB。SF path 需要保证布局正确，但主瓶颈仍是 activation materialization。

TMA col-major SF 的收益点在下游 GEMM layout；在本 scatter kernel 内，它只是一个转置式写入，不能抵消 `expanded_x` 的大流量。

## 6. 优化方案

### P0：消除 materialized expand，直接让 grouped GEMM gather `x[token]`

这是最高优先级。把 routed position 到 token 的映射暴露给 grouped GEMM：

```text
pos_to_token[p] = token_id
pos_to_expert[p] = expert_id
true_offsets[e], true_sizes[e] 描述每个 expert 的 tight row range
```

Grouped GEMM 的 A operand 从：

```text
A = expanded_x[p, k]
```

改为：

```text
token = pos_to_token[p]
A = x[token, k]
```

预期收益：

- 删除 `_expand_to_fused_main_kernel` 的 launch。
- 删除 `expanded_x` 的全量 global write。
- 删除下游 GEMM 对 `expanded_x` 的全量 global read。
- 对 H=7168 / X=4096~32768 这种远超 L2 的 shape，收益比继续调 `_NUM_THREADS` 更有上限。

风险与验证：

- GEMM 内 gather A 会让 A operand 的 token row 访问不再严格连续，需要确认 L2/Dnoc 行为。
- 如果 routing 不提供 expert-major tight layout，需要在 router/permute 阶段生成 `pos_to_token` 和 offsets。
- 必须在 64GB HBM C500 机器上做 integrated MoE benchmark；standalone `expand_to_fused` benchmark 不能代表该路径。

### P1：materialized fallback 改为 output-row-centric

如果 API 仍必须返回 `expanded_x`，建议增加一个 output-row-centric fallback：

```text
for p in expanded rows:
    token = pos_to_token[p]  # 或 dense path 下 token = p // K
    if token < 0: zero row
    else: copy x[token, :] -> expanded_x[p, :]
```

适用条件：

- router 能提供 `pos_to_token[X]`；或 benchmark/workload 明确是 dense sequential routing。
- 仍需支持 arbitrary dropped slot 和 tail zero-fill。

收益：

- 去掉 token-centric `for k in T.serial(num_topk)`。
- 对 dense routing 可跳过 `token_topk_to_pos` 读取。
- output row 连续写，更贴近目标 layout。

限制：

- 仍然 materialize `[X, H]`，无法解决下游重新读 `expanded_x` 的系统性成本。
- 若在 kernel 内动态检测 dense routing，会引入额外扫描成本；dense path 应由调用方/manifest 显式选择。

### P2：验证更宽搬运和 transaction 利用率

C500 每 16 lane 聚合 128B transaction。当前 TileLang 写法按 element 维度 `T.Parallel(hidden_aligned)` 搬运，是否充分利用 128B transaction 需要看生成代码和 profile counter。

候选方向：

- 尝试 vectorized copy（例如每 lane 搬多个连续元素），让 16-lane group 更接近 128B useful payload。
- 对 H=7168/H=3072 分别测试 `_NUM_THREADS = 64/128/256`，记录 register、private memory、kernel body 和端到端 latency。
- 对 small batch 和 prefill 分开定策略：small batch 优先融合/graph，prefill 才考虑更宽 block copy。

注意：这类优化只能改进 standalone scatter body，不应替代 P0。

### P3：memory policy / cache pollution 评估

C500 指南提到不同 CSET 可能改变 L2 行为：默认 `RWK_C` 读写检查 L2，`UNK` 读写绕过 L2，`ROK_C` 读检查 L2、写直接到内存控制器。若运行时/allocator 可以控制 CSET，可评估：

- `expanded_x` 是 streaming write，且 footprint 远超 8MB L2，写 allocate 可能污染 L2。
- 下游 GEMM 很难完全复用这份 output；对大 shape，绕过或减少写分配可能有利。
- 该项必须在远端 C500 上单独实验，且每次只改一个变量。

## 7. 测试与 benchmark 迁移清单

### 7.1 correctness

必须覆盖：

- 非 SF：bf16、fp16。
- SF：fp8 + fp32 row-major SF；fp8 + int32 packed UE8M0 TMA col-major SF；fp4 byte storage path。
- `X > T*K` padded capacity，tail rows `pos_to_expert < 0` 清零。
- `token_topk_to_pos < 0` dropped slot。
- `H` / `S` 非 64 对齐，确认 tail predicate 行为。
- TMA col-major 返回 view 的 shape、stride 和数值。

### 7.2 benchmark/profile

性能结论必须来自目标 64GB HBM C500，而不是本地机器。

建议记录：

| 类别 | 指标 | 目的 |
| --- | --- | --- |
| latency | CUPTI 或目标环境标准 timing | apples-to-apples 比较 |
| kernel body | mcTracer `_expand_to_fused_main_kernel` | 区分 body 改善和 launch overhead |
| memory | global read/write bytes、L2 hit、Dnoc latency | 判断是否 DRAM-bound |
| occupancy | workgroups/waves、register/private memory | 判断是否 block/thread 配置问题 |
| pipeline | integrated MoE trace | 确认是否真正删除 expand launch 和 materialization |

### 7.3 one-change method

每次实验只改一个变量：

| 实验 | 变量 | 保持不变 | 成功标准 |
| --- | --- | --- | --- |
| thread count | 64 -> 128 -> 256 | routing、dtype、shape、timing 协议 | latency 稳定下降且无小 batch 回退 |
| output-row fallback | token-centric -> pos_to_token | materialized API | 大 shape 两位数百分比改善 |
| dense fast path | generic routing -> dense flag | 同 shape/dtype | 跳过 routing read 后 latency 降低 |
| fused GEMM | materialized -> gather A | 同完整 MoE pipeline | trace 中无独立 expand kernel，pipeline latency 下降 |
| CSET/memory policy | RWK_C / UNK / ROK_C | 同 kernel | L2 pollution 或 DRAM traffic 改善 |

## 8. 迁移完成标准

- Op 层 contract 与上游/manifest 一致：shape、dtype、layout、sentinel 行为明确。
- Kernel 层没有把动态 batch 误放入 cache key。
- 所有 dropped/unassigned/tail/SF layout case 都有 correctness 覆盖。
- `eval_roofline()` 字节模型与 benchmark report 口径一致。
- C500 profile 能解释性能瓶颈：小 batch 是 launch/occupancy，prefill 是 DRAM materialization。
- 若目标是端到端 MoE 加速，必须提交 integrated pipeline profile，不能只提交 standalone scatter benchmark。

## 9. 当前建议

短期保留当前通用 token-centric kernel，作为语义完整的 fallback。下一阶段不要继续只围绕 `_NUM_THREADS=64` 做微调；优先实现并验证 `pos_to_token` 驱动的 output-row fallback，随后推进 grouped GEMM 直接 gather `x[token]` 的 fused 路径。C500 的 8MB L2 和 24MiB~448MiB 级别 `expanded_x` footprint 决定了：消除 materialization 的收益上限明显高于优化 standalone scatter。
