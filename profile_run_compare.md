# TileOPs Benchmark Report
Generated: 2026-08-04 11:09:18

## Environment

- **Torch version**: 2.8.0+metax3.7.1.3
- **CUDA version (torch)**: 11.6
- **GPU model**: MetaX C500
- **Driver version**: N/A

## MoeExpandToFusedFwdOp

### tileops

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.0067 | 0.0000 | 0.0366 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.0071 | 0.0000 | 0.5825 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.0448 | 0.0000 | 1.4751 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 0.4093 | 0.0000 | 1.2919 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.0055 | 0.0000 | 0.0189 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.0059 | 0.0000 | 0.2996 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.0198 | 0.0000 | 1.4331 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 0.2086 | 0.0000 | 1.0868 |

### torch-ref

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.1268 | 0.0000 | 0.0019 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.1694 | 0.0000 | 0.0244 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.7377 | 0.0000 | 0.0896 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 4.8587 | 0.0000 | 0.1088 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.1233 | 0.0000 | 0.0008 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.1484 | 0.0000 | 0.0119 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.4122 | 0.0000 | 0.0688 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 2.1592 | 0.0000 | 0.1050 |

## MoeExpandToFusedWithSFFwdOp

### tileops

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0072 | 0.0000 | 0.0175 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0291 | 0.0000 | 1.1734 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0062 | 0.0000 | 0.0199 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0349 | 0.0000 | 0.9545 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0069 | 0.0000 | 0.1335 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0055 | 0.0000 | 0.1628 |

### torch-ref

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.2445 | 0.0000 | 0.0005 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 1.1788 | 0.0000 | 0.0289 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.2414 | 0.0000 | 0.0005 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 1.1835 | 0.0000 | 0.0282 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.2797 | 0.0000 | 0.0033 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.2934 | 0.0000 | 0.0030 |

# Optimized TileOPs Benchmark Report
Generated: 2026-08-05 01:25:27

## Environment

- **Torch version**: 2.8.0+metax3.7.1.3
- **CUDA version (torch)**: 11.6
- **GPU model**: MetaX C500
- **Driver version**: N/A

## MoeExpandToFusedFwdOp

### tileops

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.0066 | 0.0000 | 0.0367 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.0070 | 0.0000 | 0.5880 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.0446 | 0.0000 | 1.4809 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 0.4092 | 0.0000 | 1.2921 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.0056 | 0.0000 | 0.0186 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.0058 | 0.0000 | 0.3051 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.0197 | 0.0000 | 1.4417 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 0.2081 | 0.0000 | 1.0897 |

### torch-ref

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.0937 | 0.0000 | 0.0026 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.1393 | 0.0000 | 0.0297 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.7028 | 0.0000 | 0.0940 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 4.8610 | 0.0000 | 0.1088 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.0914 | 0.0000 | 0.0011 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.1165 | 0.0000 | 0.0152 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.3704 | 0.0000 | 0.0765 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 2.1248 | 0.0000 | 0.1067 |

## MoeExpandToFusedWithSFFwdOp

### tileops

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0074 | 0.0000 | 0.0170 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0292 | 0.0000 | 1.1660 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0061 | 0.0000 | 0.0200 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0337 | 0.0000 | 0.9900 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0068 | 0.0000 | 0.1337 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0055 | 0.0000 | 0.1628 |

### torch-ref

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.1838 | 0.0000 | 0.0007 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 1.1153 | 0.0000 | 0.0306 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.1780 | 0.0000 | 0.0007 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 1.0806 | 0.0000 | 0.0308 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.2152 | 0.0000 | 0.0042 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.2128 | 0.0000 | 0.0042 |