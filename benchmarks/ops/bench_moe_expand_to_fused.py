"""Benchmark for MoE expand-to-fused scatter ops.

Benchmarks:
  - MoeExpandToFusedFwdOp: bf16/fp16 activation scatter into fused layout.
  - MoeExpandToFusedWithSFFwdOp: fp8 activation scatter with scale factors.

Baselines:
  - PyTorch reference: vectorized scatter into the same expanded layout.

Real model configurations in the manifest cover Kimi K2 and Qwen3-30B shapes.
"""

from typing import Any

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.moe import MoeExpandToFusedFwdOp, MoeExpandToFusedWithSFFwdOp
from workloads.workload_base import WorkloadBase

_EXPAND_OP_NAME = "MoeExpandToFusedFwdOp"
_EXPAND_WITH_SF_OP_NAME = "MoeExpandToFusedWithSFFwdOp"
_UE8M0_PER_INT32 = 4


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def _make_routing(
    total_tokens: int,
    top_k: int,
    num_expanded_tokens: int,
    *,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic routing with extra capacity marked unassigned."""
    token_topk_to_pos = torch.full(
        (total_tokens, top_k), -1, dtype=torch.int32, device=device
    )
    pos_to_expert = torch.full(
        (num_expanded_tokens,), -1, dtype=torch.int32, device=device
    )

    slot = 0
    for t in range(total_tokens):
        for k in range(top_k):
            if slot >= num_expanded_tokens:
                return token_topk_to_pos, pos_to_expert
            token_topk_to_pos[t, k] = slot
            pos_to_expert[slot] = k
            slot += 1
    return token_topk_to_pos, pos_to_expert


def _infer_sf_dtype(
    hidden_size: int,
    hidden_sf: int,
    num_per_channels: int,
) -> torch.dtype:
    """Infer float32 vs packed UE8M0 int32 from the manifest SF width."""
    sf_blocks = _ceil_div(hidden_size, num_per_channels)
    if hidden_sf == _ceil_div(sf_blocks, _UE8M0_PER_INT32):
        return torch.int32
    if hidden_sf == sf_blocks:
        return torch.float32
    raise ValueError(
        f"Cannot infer scale-factor dtype for hidden_size={hidden_size}, "
        f"hidden_sf={hidden_sf}, num_per_channels={num_per_channels}"
    )


def _torch_expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    expanded_x = torch.zeros(
        (pos_to_expert.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
    )

    flat_pos = token_topk_to_pos.reshape(-1).to(torch.int64)
    valid = flat_pos >= 0
    token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(
        token_topk_to_pos.shape[1]
    )
    expanded_x[flat_pos[valid]] = x[token_ids[valid]]
    return expanded_x


def _torch_expand_to_fused_with_sf(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_x = _torch_expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    if use_tma_aligned_col_major_sf:
        num_expanded_sf_tokens = _ceil_div(pos_to_expert.shape[0], 4) * 4
        expanded_x_sf = torch.zeros(
            (x_sf.shape[1], num_expanded_sf_tokens),
            dtype=x_sf.dtype,
            device=x_sf.device,
        )[:, : pos_to_expert.shape[0]].T
    else:
        expanded_x_sf = torch.zeros(
            (pos_to_expert.shape[0], x_sf.shape[1]),
            dtype=x_sf.dtype,
            device=x_sf.device,
        )

    flat_pos = token_topk_to_pos.reshape(-1).to(torch.int64)
    valid = flat_pos >= 0
    token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(
        token_topk_to_pos.shape[1]
    )
    expanded_x_sf[flat_pos[valid]] = x_sf[token_ids[valid]]
    return expanded_x, expanded_x_sf


class MoeExpandToFusedBench(WorkloadBase):
    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        num_expanded_tokens: int,
        dtype: torch.dtype,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_expanded_tokens = num_expanded_tokens
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, int]:
        return (self.total_tokens, self.hidden_size)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(
            self.total_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device="cuda",
        )
        token_topk_to_pos, pos_to_expert = _make_routing(
            self.total_tokens,
            self.top_k,
            self.num_expanded_tokens,
        )
        return x, token_topk_to_pos, pos_to_expert


class MoeExpandToFusedWithSFBench(WorkloadBase):
    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        hidden_sf: int,
        num_expanded_tokens: int,
        num_per_channels: int,
        dtype: torch.dtype,
        sf_dtype: torch.dtype,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.hidden_sf = hidden_sf
        self.num_expanded_tokens = num_expanded_tokens
        self.num_per_channels = num_per_channels
        self.dtype = dtype
        self.sf_dtype = sf_dtype

    @property
    def shape(self) -> tuple[int, int]:
        return (self.total_tokens, self.hidden_size)

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randint(
            -16,
            16,
            (self.total_tokens, self.hidden_size),
            dtype=torch.int8,
            device="cuda",
        ).to(self.dtype)
        if self.sf_dtype == torch.float32:
            x_sf = torch.randn(
                self.total_tokens,
                self.hidden_sf,
                dtype=self.sf_dtype,
                device="cuda",
            )
        else:
            x_sf = torch.randint(
                0,
                0x7F7F7F7F,
                (self.total_tokens, self.hidden_sf),
                dtype=self.sf_dtype,
                device="cuda",
            )
        token_topk_to_pos, pos_to_expert = _make_routing(
            self.total_tokens,
            self.top_k,
            self.num_expanded_tokens,
        )
        return x, x_sf, token_topk_to_pos, pos_to_expert


def _expand_manifest_params() -> list[Any]:
    params = []
    for workload in load_workloads(_EXPAND_OP_NAME):
        label = workload.get("label", "unlabeled")
        total_tokens, hidden_size = workload["x_shape"]
        topk_tokens, top_k = workload["token_topk_to_pos_shape"]
        assert topk_tokens == total_tokens
        (num_expanded_tokens,) = workload["pos_to_expert_shape"]
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    total_tokens,
                    top_k,
                    hidden_size,
                    num_expanded_tokens,
                    dtype,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _expand_with_sf_manifest_params() -> list[Any]:
    params = []
    for workload in load_workloads(_EXPAND_WITH_SF_OP_NAME):
        label = workload.get("label", "unlabeled")
        total_tokens, hidden_size = workload["x_shape"]
        sf_tokens, hidden_sf = workload["x_sf_shape"]
        topk_tokens, top_k = workload["token_topk_to_pos_shape"]
        assert sf_tokens == total_tokens
        assert topk_tokens == total_tokens
        (num_expanded_tokens,) = workload["pos_to_expert_shape"]
        num_per_channels = workload["num_per_channels"]
        use_tma_aligned_col_major_sf = workload["use_tma_aligned_col_major_sf"]
        sf_dtype = _infer_sf_dtype(hidden_size, hidden_sf, num_per_channels)
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    total_tokens,
                    top_k,
                    hidden_size,
                    hidden_sf,
                    num_expanded_tokens,
                    num_per_channels,
                    use_tma_aligned_col_major_sf,
                    dtype,
                    sf_dtype,
                    id=f"{label}-{dtype_str}-{str(sf_dtype).removeprefix('torch.')}",
                )
            )
    return params


@pytest.mark.parametrize(
    "total_tokens, top_k, hidden_size, num_expanded_tokens, dtype",
    _expand_manifest_params(),
)
def test_moe_expand_to_fused_bench(
    total_tokens: int,
    top_k: int,
    hidden_size: int,
    num_expanded_tokens: int,
    dtype: torch.dtype,
) -> None:
    test = MoeExpandToFusedBench(
        total_tokens,
        top_k,
        hidden_size,
        num_expanded_tokens,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MoeExpandToFusedFwdOp(dtype=dtype)
    bm = ManifestBenchmark(_EXPAND_OP_NAME, op, test)
    op(*inputs)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    _torch_expand_to_fused(*inputs)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(_torch_expand_to_fused, *inputs)
    BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")


@pytest.mark.parametrize(
    "total_tokens, top_k, hidden_size, hidden_sf, num_expanded_tokens, "
    "num_per_channels, use_tma_aligned_col_major_sf, dtype, sf_dtype",
    _expand_with_sf_manifest_params(),
)
def test_moe_expand_to_fused_with_sf_bench(
    total_tokens: int,
    top_k: int,
    hidden_size: int,
    hidden_sf: int,
    num_expanded_tokens: int,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    dtype: torch.dtype,
    sf_dtype: torch.dtype,
) -> None:
    test = MoeExpandToFusedWithSFBench(
        total_tokens,
        top_k,
        hidden_size,
        hidden_sf,
        num_expanded_tokens,
        num_per_channels,
        dtype,
        sf_dtype,
    )
    inputs = test.gen_inputs()

    op = MoeExpandToFusedWithSFFwdOp(
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        dtype=dtype,
        sf_dtype=sf_dtype,
    )
    bm = ManifestBenchmark(_EXPAND_WITH_SF_OP_NAME, op, test)
    op(*inputs)  # warmup / JIT compile
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def torch_ref_with_sf(*args):
        return _torch_expand_to_fused_with_sf(
            *args,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        )

    torch_ref_with_sf(*inputs)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(torch_ref_with_sf, *inputs)
    BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
