# Moe expand_to_fused 优化与 Benchmark 记录

## 结论

- 当前 `expand_to_fused` 是纯 scatter/copy 算子，FLOPs 为 0，性能主要受每个 workgroup 内的内存搬运并行度、写带宽和小 batch 下的 AP 利用率限制。
- `profile_files/output20260804103409/1__expand_to_fused_main_kernel.txt.csv` 显示单 kernel 只有 16 个 workgroups / waves，AP busy duty 0.15%，instruction throughput efficiency 0.03%，MMA duty 0.0%，MTE duty 1.23%，属于低并行度的搬运型瓶颈。
- 本次优化把默认 block threads 从 64 提升到 256；对非 SF 且 `hidden <= 4096` 的 prefill 大 batch 保留 128 threads，避免 3072-hidden prefill case 的 occupancy/调度退化。
- 优化未改变路由语义：仍保留 dropped slot 跳过、`pos_to_expert < 0` 的 unassigned row 置零、row-major/TMA-aligned SF 两种布局。

## 修改范围

- `tileops/kernels/moe/expand_to_fused.py`
  - 新增 `select_expand_to_fused_num_threads(hidden, num_tokens, has_scale_factors)`。
  - `_expand_to_fused_kernel(..., num_threads)` 使用动态选择的 threads 编译不同 kernel variant。
  - 默认使用 256 threads；非 SF、small-hidden、prefill 使用 128 threads。
- `tileops/ops/moe/expand_to_fused.py`
  - kernel cache key 纳入 `num_threads`，forward 时按实际 `x.shape[0]` 和 `hidden` 选择 variant。
- `tests/ops/test_moe_expand_to_fused.py`
  - 修正 SF packed/TMA reference 构造，确保 `use_tma_aligned_col_major_sf` 传入 reference test case。

## Benchmark 方法

- 设备：MetaX C500
- Torch：2.8.0+metax3.7.1.3
- CUDA version reported by torch：11.6
- benchmark 命令：
  ```bash
  PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH \
    python -m pytest -s -q benchmarks/ops/bench_moe_expand_to_fused.py
  ```
- benchmark 协议：`benchmarks/benchmark_base.py` 的 CUPTI kernel timing，10 warmup，50 repeats × 3 trials，L2 flush，取 trial mean 的 median。
- mcProfiler 基线：`profile_files/output20260804103409`。其中 `profile_files/profile_run.log` 是 mcProfiler 环境下 cuda-events timing；下面“优化前后性能”采用 `profile_run_version.md` 的 CUPTI 旧版 64-thread baseline 做 apples-to-apples 对比。

## mcProfiler 瓶颈分析

| 指标 | 单 kernel 观测 |
| --- | ---: |
| WORKGROUPS / WAVES | 16 / 16 |
| AP busy duty | 0.15% |
| instruction throughput efficiency | 0.03% |
| AP MTE / STE / MMA duty | 1.23% / 0.46% / 0.0% |
| Global read / write bytes | 22,016 / 243,936 bytes |
| L2C hit rate | 39.39% |
| Dnoc read avg latency | 197.52 cycles |
| ISU stall | vls_wdata_stall 3,808; vls_pipeline_stall 2,149 |

判断：
- 无 MMA，且 FLOPs=0；不是 compute/GEMM 问题。
- 写流量约为读流量 11.1x，符合 `x[t]` 被 broadcast 到 top-k expanded rows 的 scatter 特征。
- AP busy 极低、waves 极少，64-thread block 对 decode/small batch 和长 hidden row 的搬运并行度不足。
- 当前 kernel 仍需覆盖 `max(num_tokens, num_expanded_tokens)` 以 zero-fill unassigned rows；因此优化优先选择不改变语义的 per-block 并行度调优，而不是跳过 zero-fill 或假设 dense routing。

## Roofline

Manifest 公式：
- 非 SF：`bytes = T * H * elem_bytes + T * K * 4 + X * 4 + X * H * elem_bytes`
- SF：`bytes = T * H * elem_bytes + T * S * 4 + T * K * 4 + X * 4 + X * H * elem_bytes + X * S * 4`
- FLOPs：`0`

mcProfiler report 给出的理论带宽 `MAX_Bandwith = 1843.2 GB/s`。优化后代表性 achieved bandwidth：
| Case | Achieved TB/s | Theoretical TB/s | Achieved / theoretical |
| --- | ---: | ---: | ---: |
| non-SF H=7168 T=512 | 1.4751 | 1.8432 | 80.0% |
| non-SF H=7168 T=4096 | 1.2919 | 1.8432 | 70.1% |
| non-SF H=3072 T=512 | 1.4331 | 1.8432 | 77.8% |
| non-SF H=3072 T=4096 | 1.0868 | 1.8432 | 59.0% |
| SF row-major H=7168 T=512 | 1.1734 | 1.8432 | 63.7% |
| SF packed TMA H=7168 T=512 | 0.9545 | 1.8432 | 51.8% |

## 优化前后性能

### MoeExpandToFusedFwdOp

| T | K | H | X | dtype | baseline ms | optimized ms | speedup | optimized TB/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 7168 | 16 | bf16 | 0.0116 | 0.0067 | 1.73x | 0.0366 |
| 32 | 8 | 7168 | 256 | bf16 | 0.0118 | 0.0071 | 1.66x | 0.5825 |
| 512 | 8 | 7168 | 4096 | bf16 | 0.0853 | 0.0448 | 1.90x | 1.4751 |
| 4096 | 8 | 7168 | 32768 | bf16 | 0.7029 | 0.4093 | 1.72x | 1.2919 |
| 1 | 8 | 3072 | 16 | bf16 | 0.0067 | 0.0055 | 1.22x | 0.0189 |
| 32 | 8 | 3072 | 256 | bf16 | 0.0070 | 0.0059 | 1.19x | 0.2996 |
| 512 | 8 | 3072 | 4096 | bf16 | 0.0259 | 0.0198 | 1.31x | 1.4331 |
| 4096 | 8 | 3072 | 32768 | bf16 | 0.2309 | 0.2086 | 1.11x | 1.0868 |

### MoeExpandToFusedWithSFFwdOp

| T | K | H | S | X | SF layout | sf dtype | baseline ms | optimized ms | speedup | optimized TB/s |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 7168 | 56 | 16 | row-major | fp32 | 0.0119 | 0.0072 | 1.65x | 0.0175 |
| 512 | 8 | 7168 | 56 | 4096 | row-major | fp32 | 0.0296 | 0.0291 | 1.02x | 1.1734 |
| 1 | 8 | 7168 | 14 | 16 | TMA col-major | int32 | 0.0100 | 0.0062 | 1.61x | 0.0199 |
| 512 | 8 | 7168 | 14 | 4096 | TMA col-major | int32 | 0.0380 | 0.0349 | 1.09x | 0.9545 |
| 32 | 8 | 3072 | 24 | 256 | row-major | fp32 | 0.0084 | 0.0069 | 1.22x | 0.1335 |
| 32 | 8 | 3072 | 6 | 256 | TMA col-major | int32 | 0.0066 | 0.0055 | 1.20x | 0.1628 |

## 验证

```bash
PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH \
  python -m pytest -q tests/ops/test_moe_expand_to_fused.py -x
# 20 passed in 47.64s

PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH \
  python -m pytest -s -q benchmarks/ops/bench_moe_expand_to_fused.py
# 14 passed in 36.62s
```

## 后续建议

- 用 mcProfiler 重新采集优化后 kernel，确认 AP busy、MTE/STE duty 和 instruction throughput efficiency 是否随 256-thread variant 提升。
- 如果后续 routing contract 能显式声明 dense/no-dropped/no-padding，可新增 scatter-only fast path，跳过 `pos_to_expert < 0` 全 X 检查；当前未这样做是为了保持现有 dropped/unassigned 语义。

