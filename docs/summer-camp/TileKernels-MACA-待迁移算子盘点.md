# TileKernels-MACA-待迁移算子盘点

[**简体中文**](TileKernels-MACA-待迁移算子盘点.md) | [English](TileKernels-MACA-operator-migration-inventory.en.md)

待迁移算子的统一来源是[MetaX-MACA/TileKernels-Metax ](https://github.com/MetaX-MACA/TileKernels-Metax)默认 dev 分支。本清单仅收录经过初步评估、具有通用复用价值、MetaX C500 适配或性能优化空间，并能够形成独立测试与 Benchmark 的候选算子。

| 类别 | 算子名称 | 状态 | 难度 | 说明 |
| --- | --- | --- | --- | --- |
| Engram | engram_fused_weight | 待迁移 | 低 | 逐元素权重融合、向量化访存和 FP32 输出。 |
| Engram | engram_gate | 待迁移 | 高 | 持久化 Kernel、共享内存流水、归约、RMSNorm、门控和反向计算。 |
| Engram | engram_grad_w_reduce | 待迁移 | 低 | 权重梯度归约、流水拷贝和原地累加。 |
| Engram | engram_hash | 待迁移 | 低 | int64 哈希、位运算、取模和索引计算。 |
| Manifold HyperConnection | mhc_multilayer_recompute | 待迁移 | 高 | 设备指针、多层重计算、运行时 Tensor 构造和双缓冲。 |
| Manifold HyperConnection | mhc_post | 待优化 | 中 | 矩阵混合、前向与反向、Reducer 和流水线。 |
| Manifold HyperConnection | mhc_pre_big_fuse | 待迁移 | 高 | RMSNorm、Sigmoid、Sinkhorn、Split-K 和融合 Kernel。 |
| MoE Routing | moe_aux_fi | 待迁移 | 低 | 共享内存直方图、原子加和统计归一化。 |
| MoE Routing | moe_expand_to_fused | 待迁移 | 中 | Scatter、Expert-Major 布局、FP8/FP4 和缩放因子布局。 |
| MoE Routing | moe_get_fused_mapping | 待迁移 | 高 | Warp 直方图、Grid 同步、前缀和和稳定排名。 |
| MoE Routing | moe_inplace_unique_group_indices | 待迁移 | 低 | 位图、位运算和原地去重。 |
| MoE Routing | moe_mask_indices_by_tp | 待迁移 | 低 | TP 映射、整数索引和范围过滤。 |
| MoE Routing | moe_normalize_weight | 待迁移 | 低 | Top-K 权重归一化、行归约和数值稳定性。 |
| MoE Routing | moe_reduce_fused | 待迁移 | 中 | Gather、加权归约、FP8 输出和缩放因子。 |
| MoE Routing | moe_group_count | 待优化 | 低 | 共享内存直方图、原子操作和并行统计。 |
| MoE Gating | moe_top2_sum_gate | 待迁移 | 高 | 多种 Scoring、分组选取、稳定 Top-K、Expert 映射和 EP/TP Mask。 |
| MoE Gating | moe_topk_gate | 待迁移 | 中 | Top-K、最大值归约和稳定索引选择。 |
| MoE Gating | moe_topk_sum_group_idx | 待迁移 | 中 | 分组选择、Lane Shuffle 和 Wavefront 协作。 |
| Quantization | quant_cast_back | 待迁移 | 高 | FP8/FP4 反量化、缩放因子、布局处理和 E5M6。 |
| Quantization | quant_cast_back_e5m6 | 待迁移 | 中 | E5M6、位打包与解包和自定义 Fragment。 |
| Quantization | quant_per_block_cast | 待迁移 | 中 | 分块量化、AMax、FP8/FP4 和自定义 Layout。 |
| Quantization | quant_per_block_cast_lossless | 待迁移 | 中 | 无损再量化、指数运算、饱和处理和 Device Assert。 |
| Quantization | quant_per_channel_cast | 待迁移 | 低 | 参数校验、封装和 Kernel 调度。 |
| Quantization | quant_per_channel_cast_transpose | 待迁移 | 中 | 按通道量化、寄存器转置、Shared-Memory Swizzle 和列归约。 |
| Quantization | quant_per_channel_cast_fused | 待迁移 | 高 | 按通道量化、Gather/Expand、FP8 Rescale 和 Lane 广播。 |
| Quantization | quant_per_token_cast | 待迁移 | 高 | 逐 Token 量化、AMax、自定义 Layout、多种数值格式和 Strided 输入。 |
| Quantization | quant_per_token_cast_e5m6 | 待迁移 | 高 | E5M6 编码、位打包、舍入和底层浮点转换。 |
| Fused SwiGLU + Quantization | quant_swiglu_bwd_token_cast | 待迁移 | 高 | 反量化、SwiGLU 反向、多输出量化和梯度归约。 |
| Fused SwiGLU + Quantization | quant_swiglu_fwd_channel_cast_transpose | 待迁移 | 高 | SwiGLU 前向、按通道量化、转置和 Packed BF16。 |
| Fused SwiGLU + Quantization | quant_swiglu_fwd_token_cast | 待迁移 | 高 | SwiGLU 前向、逐 Token 量化、Mask、原子操作和持久化 Kernel。 |
| Transpose | batched_transpose | 待迁移 | 中 | 批量转置、寄存器转置、Shared-Memory Swizzle 和向量化。 |
