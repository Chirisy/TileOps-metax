# TileOPs Benchmark Report
Generated: 2026-08-04 07:45:30

## Environment

- **Torch version**: 2.8.0+metax3.7.1.3
- **CUDA version (torch)**: 11.6
- **GPU model**: MetaX C500
- **Driver version**: N/A

## MoeExpandToFusedFwdOp

### tileops

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.0116 | 0.0000 | 0.0211 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.0118 | 0.0000 | 0.3514 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.0853 | 0.0000 | 0.7747 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 0.7029 | 0.0000 | 0.7522 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.0067 | 0.0000 | 0.0156 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.0070 | 0.0000 | 0.2540 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.0259 | 0.0000 | 1.0956 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 0.2309 | 0.0000 | 0.9821 |

### torch-ref

| total_tokens | top_k | hidden_size | num_expanded_tokens | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 16 | torch.bfloat16 | 0.1195 | 0.0000 | 0.0020 |
| 32 | 8 | 7168 | 256 | torch.bfloat16 | 0.1600 | 0.0000 | 0.0258 |
| 512 | 8 | 7168 | 4096 | torch.bfloat16 | 0.7257 | 0.0000 | 0.0911 |
| 4096 | 8 | 7168 | 32768 | torch.bfloat16 | 4.8673 | 0.0000 | 0.1086 |
| 1 | 8 | 3072 | 16 | torch.bfloat16 | 0.1201 | 0.0000 | 0.0009 |
| 32 | 8 | 3072 | 256 | torch.bfloat16 | 0.1375 | 0.0000 | 0.0129 |
| 512 | 8 | 3072 | 4096 | torch.bfloat16 | 0.3945 | 0.0000 | 0.0718 |
| 4096 | 8 | 3072 | 32768 | torch.bfloat16 | 2.1523 | 0.0000 | 0.1054 |

## MoeExpandToFusedWithSFFwdOp

### tileops

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0119 | 0.0000 | 0.0105 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0296 | 0.0000 | 1.1503 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0100 | 0.0000 | 0.0123 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0380 | 0.0000 | 0.8776 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.0084 | 0.0000 | 0.1091 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.0066 | 0.0000 | 0.1364 |

### torch-ref

| total_tokens | top_k | hidden_size | hidden_sf | num_expanded_tokens | num_per_channels | use_tma_aligned_col_major_sf | dtype | sf_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | 56 | 16 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.2426 | 0.0000 | 0.0005 |
| 512 | 8 | 7168 | 56 | 4096 | 128 | False | torch.float8_e4m3fn | torch.float32 | 1.1800 | 0.0000 | 0.0289 |
| 1 | 8 | 7168 | 14 | 16 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.2354 | 0.0000 | 0.0005 |
| 512 | 8 | 7168 | 14 | 4096 | 128 | True | torch.float8_e4m3fn | torch.int32 | 1.1534 | 0.0000 | 0.0289 |
| 32 | 8 | 3072 | 24 | 256 | 128 | False | torch.float8_e4m3fn | torch.float32 | 0.2671 | 0.0000 | 0.0034 |
| 32 | 8 | 3072 | 6 | 256 | 128 | True | torch.float8_e4m3fn | torch.int32 | 0.2699 | 0.0000 | 0.0033 |
