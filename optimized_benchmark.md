# Moe expand_to_fused 64GB C500 Profile 复盘与下一版优化方案

## 结论

- `profile_run_compare.md` 是这次最可信的 apples-to-apples 对比：新旧 `tileops` latency 基本没有明显下降，非 SF 仅约 `0.0%~1.7%` 改善，SF 约 `-2.8%~+3.4%` 波动，属于噪声级。
- 当前线程数调优确实让部分 raw kernel body 变快：mcTracer 显示 H=7168、X=32768 的 `_expand_to_fused_main_kernel` 从约 `702.5us` 降到约 `408.0us`。但这没有稳定转化成 benchmark latency。
- 根因不是继续调 `num_threads` 就能解决的单点瓶颈，而是 `expand_to_fused` 的结构性成本：单独 kernel launch + materialized scatter 写出 `[X, H]` + 下游 GEMM 再读 `[X, H]`。
- 结合 C500 层级参数看，大 shape 的 `expanded_x` 远超 8MB L2：H=7168、X=4096 是 56MiB，H=7168、X=32768 是 448MiB；下游 GEMM 基本不能指望从 L2 复用这份 materialized output。
- 下一版优化优先级应从“更快 scatter”转为“减少/消除 scatter”：首选把 expand 与 grouped GEMM 消费端融合，或让 GEMM 通过 routing map 直接 gather 原始 `x[token]`。

本地机器不是 64GB HBM C500，且用户明确说明本地 benchmark / mcProfile / mcTracer 参考性不大；本文只使用 `profile/`、`profile_old/` 和 `profile_run_compare.md` 的远端结果。

## 数据来源

- 新版本 profile：`profile/`
- benchmark：`profile/profile_run.log`
- tracer：`profile/tracer_out_20260805013109/tracer_out-623632.json`
- mcProfiler bottleneck json：`profile/output20260805013430/moe_expand_to_fused_all_no_sf_and_with_sf_bottleneck.json`
- 旧版本 profile：`profile_old/`
- benchmark：`profile_old/profile_run.log`
- tracer：`profile_old/tracer_out_20260804090918/tracer_out-112315.json`
- mcProfiler CSV：`profile_old/output20260804103409/1__expand_to_fused_main_kernel.txt.csv`
- 新旧 CUPTI benchmark 对比：`profile_run_compare.md`
- C500 层级参数：L2 8MB，SM/shared memory 64KB，vector L1 32KB，constant L1 8KB，vector register file `64 lanes * 2048 * 4B = 512KB`，scalar register file `800 * 4B`

`profile/output20260805013430/moe_expand_to_fused_all_no_sf_and_with_sf_bottleneck.json` 这次没有有效 perf counter，只有 device info；因此新的深层分析主要依赖 mcTracer，旧版深层 counter 使用 `profile_old/output20260804103409/1__expand_to_fused_main_kernel.txt.csv`。

## C500 层级参数对瓶颈的影响

| 项目 | 容量 / 规模 | 对 `expand_to_fused` 的含义 |
| --- | ---: | --- |
| L2 cache | 8MB | 只能容纳小 batch 或单个输入 `x` 的一部分，无法容纳 prefill 的 materialized `expanded_x`。 |
| SM/shared memory | 64KB | H=7168 的 bf16 单行约 14KB，单行 staging 可以放下，但 top-k scatter 输出不能靠 SM 层级缓存吸收。 |
| vector L1 | 32KB | 对单行 copy 有帮助，但对 24MiB~448MiB 级别的输出流写无法形成跨 kernel 复用。 |
| constant L1 | 8KB | routing 常量/小表可能受益，但不是当前主流量。 |
| vector register file | 512KB | 新 trace 中 regs/private memory 已明显改善，说明 register 压力不是剩余端到端瓶颈。 |
| scalar register file | 3.125KB | 主要影响索引/标量控制，不改变 materialized output 的带宽上限。 |

典型 materialized output footprint：

| H | X | dtype | `expanded_x` bytes | MiB | vs 8MB L2 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 7168 | 4096 | bf16/fp16 | 58,720,256 | 56 | 7.0x |
| 7168 | 32768 | bf16/fp16 | 469,762,048 | 448 | 56.0x |
| 3072 | 4096 | bf16/fp16 | 25,165,824 | 24 | 3.0x |
| 3072 | 32768 | bf16/fp16 | 201,326,592 | 192 | 24.0x |

因此，单独优化 scatter kernel 的 register 或 block width，只能让本 kernel body 更快；它不能避免 `expanded_x` 的 global write，也不能让下游 GEMM 从 8MB L2 中复用 24MiB~448MiB 的 materialized buffer。这个层级约束进一步支持 P0：把 expand 消掉，或把下游 GEMM 改成直接按 routing gather `x[token]`。

## apples-to-apples latency 对比

`profile_run_compare.md` 中旧 report 与 optimized report 的 `tileops` 数据如下。结论是：当前优化没有产生稳定的端到端 latency 收益。

### MoeExpandToFusedFwdOp

| T | K | H | X | old ms | new ms | speedup | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7168 | 16 | 0.0067 | 0.0066 | 1.015x | -1.49% |
| 32 | 8 | 7168 | 256 | 0.0071 | 0.0070 | 1.014x | -1.41% |
| 512 | 8 | 7168 | 4096 | 0.0448 | 0.0446 | 1.004x | -0.45% |
| 4096 | 8 | 7168 | 32768 | 0.4093 | 0.4092 | 1.000x | -0.02% |
| 1 | 8 | 3072 | 16 | 0.0055 | 0.0056 | 0.982x | +1.82% |
| 32 | 8 | 3072 | 256 | 0.0059 | 0.0058 | 1.017x | -1.69% |
| 512 | 8 | 3072 | 4096 | 0.0198 | 0.0197 | 1.005x | -0.51% |
| 4096 | 8 | 3072 | 32768 | 0.2086 | 0.2081 | 1.002x | -0.24% |

### MoeExpandToFusedWithSFFwdOp

| T | K | H | H_sf | X | SF layout | old ms | new ms | speedup | delta |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 7168 | 56 | 16 | row-major | 0.0072 | 0.0074 | 0.973x | +2.78% |
| 512 | 8 | 7168 | 56 | 4096 | row-major | 0.0291 | 0.0292 | 0.997x | +0.34% |
| 1 | 8 | 7168 | 14 | 16 | TMA col-major | 0.0062 | 0.0061 | 1.016x | -1.61% |
| 512 | 8 | 7168 | 14 | 4096 | TMA col-major | 0.0349 | 0.0337 | 1.036x | -3.44% |
| 32 | 8 | 3072 | 24 | 256 | row-major | 0.0069 | 0.0068 | 1.015x | -1.45% |
| 32 | 8 | 3072 | 6 | 256 | TMA col-major | 0.0055 | 0.0055 | 1.000x | +0.00% |

## profile_old vs profile 的补充观察

`profile_old/profile_run.log` 与 `profile/profile_run.log` 是 mcProfiler/cuda-events 环境下的新旧目录对比，不能替代上面的 CUPTI apples-to-apples 结论，但能说明线程数调优的真实影响范围。

### 非 SF

| T | H | X | old ms | new ms | speedup | 观察 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 7168 | 16 | 0.0344 | 0.0527 | 0.653x | 小 batch 变慢 |
| 32 | 7168 | 256 | 0.0320 | 0.0530 | 0.604x | 小 batch 变慢 |
| 512 | 7168 | 4096 | 0.1013 | 0.0615 | 1.647x | 大 H/prefill 变快 |
| 4096 | 7168 | 32768 | 0.7188 | 0.4238 | 1.696x | 大 H/prefill 变快 |
| 1 | 3072 | 16 | 0.0336 | 0.0336 | 1.000x | 基本不变 |
| 32 | 3072 | 256 | 0.0326 | 0.0337 | 0.967x | 略慢 |
| 512 | 3072 | 4096 | 0.0424 | 0.0373 | 1.137x | 小幅变快 |
| 4096 | 3072 | 32768 | 0.2462 | 0.2237 | 1.101x | 小幅变快 |

### SF

| T | H | H_sf | X | SF layout | old ms | new ms | speedup | 观察 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 7168 | 56 | 16 | row-major | 0.0428 | 0.0409 | 1.046x | 小幅/noisy |
| 512 | 7168 | 56 | 4096 | row-major | 0.0481 | 0.0472 | 1.019x | 小幅/noisy |
| 1 | 7168 | 14 | 16 | TMA col-major | 0.0500 | 0.0499 | 1.002x | 基本不变 |
| 512 | 7168 | 14 | 4096 | TMA col-major | 0.0579 | 0.0549 | 1.055x | 小幅变快 |
| 32 | 3072 | 24 | 256 | row-major | 0.0393 | 0.0411 | 0.956x | 略慢 |
| 32 | 3072 | 6 | 256 | TMA col-major | 0.0485 | 0.0544 | 0.892x | 变慢 |

## tracer 与 mcProfiler 结论

### mcTracer kernel body

| 版本 | grid.x | block.x | regs/thread | private total | mtreg occ | mean body time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old | 16 | 64 | 110 | 228 | 21% | 11.533us |
| new | 16 | 256 | 20 | 0 | 3% | 6.660us |
| old | 4096 | 64 | 110 | 228 | 21% | 85.197us |
| new | 4096 | 256 | 20 | 0 | 3% | 44.703us |
| old | 32768 | 64 | 110 | 228 | 21% | 702.459us |
| new | 32768 | 256 | 20 | 0 | 3% | 407.987us |
| old | 32768 | 64 | 44 | 0 | 8% | 230.339us |
| new | 32768 | 128 | 24 | 0 | 4% | 208.768us |

`dur` 在 tracer JSON 中按 ns 记录，上表已换算为 us。这个结果说明上一版优化方向“降低 register/private memory、提高每个 block 的 copy 并行度”对 kernel body 是有效的，尤其是 H=7168 大 shape。

### launch / runtime overhead

新旧 trace 中 runtime launch 开销基本不变：

| event | old p50 | new p50 |
| --- | ---: | ---: |
| mcLaunchKernel | 4.526us | 4.599us |
| mcModuleLaunchKernel | 4.277us | 4.186us |

对于 T=1/32 这类小 shape，kernel body 只有约 `5~7us`，单次 launch/runtime 开销已经同量级；因此 standalone kernel 内部再省几 us，也很难在端到端 latency 上稳定显现。

### 旧版 mcProfiler counter

旧版 `1__expand_to_fused_main_kernel.txt.csv` 的关键 counter：

| 指标 | 观测 |
| --- | ---: |
| WORKGROUPS / WAVES | 16 / 16 |
| AP busy duty | 0.15% |
| instruction throughput efficiency | 0.03% |
| AP MTE / STE / MMA duty | 1.23% / 0.46% / 0.0% |
| Global read / write bytes | 22,016 / 243,936 bytes |
| L2C hit rate | 39.39% |
| Dnoc read average latency | 197.52 cycles |
| ISU stall | vls_wdata_stall 3,808; vls_pipeline_stall 2,149 |

判断：

- `expand_to_fused` FLOPs 为 0，MMA duty 为 0；瓶颈不是 compute。
- 写流量显著高于读流量，符合 `x[token]` broadcast 到 top-k expanded rows 的 scatter 模式。
- 低 AP busy + 少 waves 表明小 batch 下 occupancy/调度利用率差；但新 trace 已经证明继续调 block threads 只能改善 kernel body，不能消除 standalone kernel 的 launch 和 materialization 成本。
- C500 的 8MB L2 无法覆盖 prefill `expanded_x` footprint，因此下游 GEMM 对 `expanded_x` 的读取大概率是一次新的全局内存流量，而不是便宜的 L2 hit。
- 当前 benchmark 的 routing 是 dense sequential：`token_topk_to_pos[t,k] = t*K+k`，`pos_to_expert[slot] = k`，tail capacity 为 `-1`。通用 scatter kernel 为支持任意 routing/dropped slot 付出了不必要的索引与 top-k 循环成本。

## 当前实现的问题边界

当前 `tileops/kernels/moe/expand_to_fused.py` 是 token-centric scatter：

1. 每个 token block 先把 `x[t, :]` 载入 fragment。
2. 对 `num_topk` 串行遍历 `token_topk_to_pos[t, k]`。
3. 对每个有效 position 写出 `expanded_x[p, :]`。
4. 对 `pos_to_expert[p] < 0` 的 expanded row 执行 zero-fill。
5. SF path 额外对 `expanded_x_sf` 做同样 scatter，且 TMA col-major layout 还有转置式写入。

这个设计语义完整，适合任意 sparse/dropped routing；但它有两个无法通过 `num_threads` 根治的问题：

- 它必须 materialize `expanded_x`，写出 `X * H * elem_bytes`，随后下游 expert GEMM 还会重新读入同一份数据。
- 它以 token 为中心写 top-k 多个 row，泛化能力强，但对 benchmark 和常见 dense routing 来说，不如 output-row-centric copy 直接。

## 下一版优化方案

### P0：融合/索引化下游 grouped GEMM，消除 materialized expand

这是最应该优先做的方向。不要让 `expand_to_fused` 先生成 `[X, H]`，而是在 MoE grouped GEMM 中直接通过 routing map 读取原始 `x[token]`。

建议路径：

1. 在 routing 阶段生成或暴露 `pos_to_token` / `pos_to_token_topk`：
   - `pos_to_token[p] = token_id`
   - unassigned 或 padding position 设为 `-1`
   - 如需要 top-k 权重/专家信息，保留 `pos_to_expert` 或 `expert_offsets/true_sizes`
2. 改造 grouped GEMM consumer：
   - 对 expert-major 的 M row，先由 position 得到 `token_id`
   - A operand 不再从 `expanded_x[p, k]` 读取，而是从 `x[token_id, k]` gather
   - B operand 仍按 expert 读取权重
3. 优先复用已有 no-pad grouped GEMM 结构：
   - `tileops/kernels/moe/moe_grouped_gemm_nopad.py` 已经有 `true_offsets` / `true_sizes`
   - `tileops/kernels/moe/permute_nopad.py` 一类路径已经在处理 tight expert layout
   - 新路径应尽量接到这些结构，而不是再增加一个 standalone scatter 后端
4. 成功标准：
   - integrated MoE trace 中不再出现独立 `_expand_to_fused_main_kernel`
   - pipeline latency 至少省掉一次 launch，并减少 `expanded_x` 写 + GEMM 读的全局内存流量
   - 对 H=7168、T=512/4096 这类大 shape，应比 standalone scatter 优化更明显

预期收益来源：

- 删除一个 kernel launch。
- 删除 `expanded_x` 全量 global write。
- 删除下游 GEMM 对 `expanded_x` 的 global read；在 8MB L2 下，这部分对 24MiB~448MiB 的 prefill output 基本不能依赖 cache 复用。
- 对 SF path 可同步删除/减少 `expanded_x_sf` materialization，尤其是 packed/TMA layout 的额外写流量。

主要风险：

- GEMM 内 gather A operand 会引入非连续 token row 读取，需要确认 L2/cache 行为。
- 如果 routing 本身不按 expert-major/tight layout 输出，需要 router 提供稳定的 `pos_to_token` 和 expert offsets。
- 需要在远端 64GB C500 上做 integrated benchmark；standalone `expand_to_fused` benchmark 不再能代表该路径收益。

### P1：为 materialized fallback 增加 output-row-centric kernel

如果 API 仍必须返回 `expanded_x`，下一步不应继续微调当前 token-centric scatter，而应加一个 output-row-centric fallback。

建议新增输入或内部变体：

- 通用版本：输入 `pos_to_token[X]`，每个 output row 直接 copy `x[pos_to_token[p], :]`。
- dense benchmark 版本：当 routing 保证 `p = token * K + k` 时，不读取 `token_topk_to_pos`，直接 `token = p // K`。
- unassigned tail：`p >= T*K` 或 `pos_to_token[p] < 0` 时 zero-fill。
- SF row-major：同样按 output row copy `x_sf[token, :]`。
- SF TMA col-major：保留 column-major destination，但索引以 output position 为主，避免 token-centric top-k 串行 scatter。

为什么它比当前 fallback 更有希望：

- 去掉每个 token 内 `num_topk` 串行循环。
- 写入天然按 output row 划分，更贴近目标 layout。
- 对 dense sequential routing 可完全跳过 routing map 读取。
- 对任意 routing，只要 router 能提供 `pos_to_token`，仍保持语义完整。

限制：

- 仍然 materialize `[X, H]`，所以大 shape 的最终上限仍由全局内存流量决定。
- C500 的 L2/向量 L1 容量决定了 fallback 即使更快，也主要省 token-centric 索引与循环开销，不能解决下游重新读取 `expanded_x` 的系统性成本。
- 如果为了判断 dense routing 在 kernel 内扫描 `token_topk_to_pos`，检测成本可能抵消收益；dense fast path 应由调用方显式选择，或由 benchmark/workload manifest 明确标记。
- 需要新增 correctness tests 覆盖 dropped slot、tail zero-fill、SF row-major、SF TMA col-major。

### P2：小 batch launch-bound 路径只做融合，不再单独追求 kernel micro-optimization

对 T=1/32：

- 新 trace 中 kernel body 约 `5~7us`。
- launch/runtime p50 约 `4.2~4.6us`。
- `profile_run_compare.md` 中 T=1/32 的 latency 变化只有 `-1.7%~+2.8%`。

因此小 batch 的优化优先级应是：

1. 融合到下游 kernel。
2. 若部署形态允许，用 CUDA graph / runtime capture 降低 launch overhead。
3. 不再为 standalone `expand_to_fused` 继续增加过多 shape-specific thread heuristic。

## 建议实施顺序

1. **短期文档与验证口径**
   - 保留当前 256/128-thread 优化代码作为已验证的 kernel-body 改善，但不再宣称端到端 latency 有显著提升。
   - 后续 benchmark 报告以 `profile_run_compare.md` 风格的 CUPTI apples-to-apples 结果为主。
2. **P1 fallback 原型**
   - 新增 `pos_to_token` 或 dense flag 的 output-row-centric materialized kernel。
   - 先覆盖 standalone API，方便用现有 benchmark 验证。
   - 远端 64GB C500 成功标准：非 SF/SF 大 shape 至少有稳定两位数百分比改善，否则停止在该方向投入。
3. **P0 integrated MoE 路径**
   - 在 grouped GEMM consumer 中直接 gather `x[token]`。
   - 把 `expand_to_fused` 从完整 MoE pipeline trace 中移除。
   - 远端 64GB C500 成功标准：pipeline latency 降幅接近或超过 standalone expand 的原始占比，并确认 GEMM gather 没有把瓶颈转移到 L2/Dnoc。
4. **回归与 profile**
   - correctness：覆盖 arbitrary routing、dropped slot、tail zero-fill、row-major SF、TMA col-major SF。
   - profile：只使用远端 64GB HBM C500 数据更新本文；本地只做编译/单元测试，不做性能结论。
