"""Reference and optional kernel tests for moe_expand_to_fused.

The operator contract is a scatter-copy into the fused expert layout:
  - token_topk_to_pos[t, k] >= 0 copies x[t] to output[token_topk_to_pos[t, k]]
  - token_topk_to_pos[t, k] < 0 is skipped
  - pos_to_expert[pos] < 0 marks padding; padding output rows stay zero
  - no weights, sums, or reductions are applied

The cases mirror the TileKernels-Metax source implementation at
79461d72d6f97f91b89403705a841c9d4c1eddd6:
  - base expand_to_fused for bf16/fp16 activation rows
  - expand_to_fused_with_sf for fp8 activations plus row-major fp32 or packed
    int32 scale factors
  - TMA-aligned scale factors represented publicly as shape [X, S] with a
    column-major stride

Most tests exercise the pure-PyTorch reference so they can run without a C500
runtime. CUDA smoke tests import TileLang-backed modules lazily; they compile
and execute the new kernel only when CUDA and the MACA TileLang build are
available.
"""

import os

import pytest
import torch

os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
_DOCUMENTED_HIDDEN_SIZES = [576, 2048, 2560, 3072, 4096, 6144, 7168]
_DOCUMENTED_TOPK = [2, 6, 8, 9]
_DOCUMENTED_NUM_PER_CHANNELS = [32, 128]

_BASE_MANIFEST_WORKLOADS = [
    pytest.param(
        "kimi-k2-decode",
        1,
        7168,
        8,
        16,
        marks=pytest.mark.smoke,
        id="kimi-k2-decode",
    ),
    pytest.param(
        "kimi-k2-small",
        32,
        7168,
        8,
        256,
        marks=pytest.mark.full,
        id="kimi-k2-small",
    ),
    pytest.param(
        "kimi-k2-medium",
        512,
        7168,
        8,
        4096,
        marks=pytest.mark.full,
        id="kimi-k2-medium",
    ),
    pytest.param(
        "kimi-k2-prefill",
        4096,
        7168,
        8,
        32768,
        marks=pytest.mark.full,
        id="kimi-k2-prefill",
    ),
    pytest.param(
        "qwen3-30b-decode",
        1,
        3072,
        8,
        16,
        marks=pytest.mark.full,
        id="qwen3-30b-decode",
    ),
    pytest.param(
        "qwen3-30b-small",
        32,
        3072,
        8,
        256,
        marks=pytest.mark.full,
        id="qwen3-30b-small",
    ),
    pytest.param(
        "qwen3-30b-medium",
        512,
        3072,
        8,
        4096,
        marks=pytest.mark.full,
        id="qwen3-30b-medium",
    ),
    pytest.param(
        "qwen3-30b-prefill",
        4096,
        3072,
        8,
        32768,
        marks=pytest.mark.full,
        id="qwen3-30b-prefill",
    ),
]

_SF_MANIFEST_WORKLOADS = [
    pytest.param(
        "kimi-k2-decode-sf-row-major",
        1,
        7168,
        8,
        16,
        128,
        False,
        torch.float32,
        False,
        marks=pytest.mark.smoke,
        id="kimi-k2-decode-row-major",
    ),
    pytest.param(
        "kimi-k2-medium-sf-row-major",
        512,
        7168,
        8,
        4096,
        128,
        False,
        torch.float32,
        False,
        marks=pytest.mark.full,
        id="kimi-k2-medium-row-major",
    ),
    pytest.param(
        "kimi-k2-decode-sf-packed",
        1,
        7168,
        8,
        16,
        128,
        True,
        torch.int32,
        True,
        marks=pytest.mark.full,
        id="kimi-k2-decode-packed",
    ),
    pytest.param(
        "kimi-k2-medium-sf-packed",
        512,
        7168,
        8,
        4096,
        128,
        True,
        torch.int32,
        True,
        marks=pytest.mark.full,
        id="kimi-k2-medium-packed",
    ),
    pytest.param(
        "qwen3-30b-small-sf-row-major",
        32,
        3072,
        8,
        256,
        128,
        False,
        torch.float32,
        False,
        marks=pytest.mark.full,
        id="qwen3-30b-small-row-major",
    ),
    pytest.param(
        "qwen3-30b-small-sf-packed",
        32,
        3072,
        8,
        256,
        128,
        True,
        torch.int32,
        True,
        marks=pytest.mark.full,
        id="qwen3-30b-small-packed",
    ),
]

def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def _align(x: int, alignment: int) -> int:
    return _ceil_div(x, alignment) * alignment

def _is_float8_dtype(dtype: torch.dtype) -> bool:
    fp8_dtypes = {
        dt
        for dt in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dt is not None
    }
    return dtype in fp8_dtypes

def _sf_channels(hidden: int, num_per_channels: int, *, packed: bool) -> int:
    channels = _ceil_div(hidden, num_per_channels)
    return _ceil_div(channels, 4) if packed else channels

def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if _is_float8_dtype(actual.dtype):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    elif actual.is_floating_point():
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert torch.equal(actual, expected)

def _make_x(num_tokens: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(num_tokens * hidden, dtype=torch.float32).view(num_tokens, hidden)
    if _is_float8_dtype(dtype):
        return ((values.remainder(32) - 16) / 8).to(dtype)
    return (values / 100).to(dtype)

def _make_scale_factors(num_tokens: int, channels: int, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(num_tokens * channels, dtype=torch.float32).view(num_tokens, channels)
    if dtype == torch.int32:
        return (values.to(torch.int32) + 1) * 17
    return (values / 100 + 1).to(dtype)

def _source_rows(token_topk_to_pos: torch.Tensor) -> torch.Tensor:
    return torch.arange(
        token_topk_to_pos.shape[0],
        dtype=torch.int64,
        device=token_topk_to_pos.device,
    ).view(-1, 1).expand_as(token_topk_to_pos)

def _validate_expand_inputs(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape={tuple(x.shape)}")
    if token_topk_to_pos.ndim != 2:
        raise ValueError(
            f"token_topk_to_pos must be 2-D, got shape={tuple(token_topk_to_pos.shape)}"
        )
    if token_topk_to_pos.shape[0] != x.shape[0]:
        raise ValueError(
            "token_topk_to_pos.shape[0] must match x.shape[0], "
            f"got {token_topk_to_pos.shape[0]} and {x.shape[0]}"
        )
    if pos_to_expert.ndim != 1:
        raise ValueError(f"pos_to_expert must be 1-D, got shape={tuple(pos_to_expert.shape)}")
    if token_topk_to_pos.dtype != torch.int32:
        raise ValueError(f"token_topk_to_pos must be int32, got {token_topk_to_pos.dtype}")
    if pos_to_expert.dtype != torch.int32:
        raise ValueError(f"pos_to_expert must be int32, got {pos_to_expert.dtype}")

    # These malformed-mapping checks keep synthetic test inputs unambiguous.
    # They are reference sanity checks, not a requirement that the eventual
    # kernel performs runtime validation for undefined inputs.
    valid = token_topk_to_pos >= 0
    if valid.any():
        valid_pos = token_topk_to_pos[valid]
        if valid_pos.max().item() >= pos_to_expert.numel():
            raise ValueError("token_topk_to_pos contains a position outside pos_to_expert")
        if torch.unique(valid_pos).numel() != valid_pos.numel():
            raise ValueError("token_topk_to_pos contains duplicate valid positions")
        if (pos_to_expert[valid_pos.long()] < 0).any():
            raise ValueError("token_topk_to_pos points to a padding position")

def _ref_moe_expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for MoeExpandToFusedFwdOp."""
    _validate_expand_inputs(x, token_topk_to_pos, pos_to_expert)

    out = torch.zeros(
        (pos_to_expert.numel(), x.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    valid = token_topk_to_pos >= 0
    if valid.any():
        src_rows = _source_rows(token_topk_to_pos)
        dst_rows = token_topk_to_pos[valid].long()
        src_rows = src_rows[valid].long()
        if _is_float8_dtype(x.dtype):
            # PyTorch's CPU float8 tensors do not implement advanced indexing
            # (`index_cpu`), so copy the exact encoded bytes for reference tests.
            out.view(torch.uint8)[dst_rows] = x.view(torch.uint8)[src_rows]
        else:
            out[dst_rows] = x[src_rows]
    return out

def _ref_moe_expand_to_fused_with_sf(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for MoeExpandToFusedWithSFFwdOp."""
    if num_per_channels not in _DOCUMENTED_NUM_PER_CHANNELS:
        raise ValueError(f"unsupported num_per_channels={num_per_channels}")
    if use_packed_ue8m0 and not use_tma_aligned_col_major_sf:
        raise ValueError("packed UE8M0 scale factors require TMA column-major layout")
    if x_sf.ndim != 2:
        raise ValueError(f"x_sf must be 2-D, got shape={tuple(x_sf.shape)}")

    expected_s = _sf_channels(
        x.shape[1],
        num_per_channels,
        packed=use_packed_ue8m0,
    )
    if x_sf.shape != (x.shape[0], expected_s):
        raise ValueError(
            "x_sf must have shape (T, S), where S follows the manifest scale-factor "
            f"rule; got {tuple(x_sf.shape)} for x.shape={tuple(x.shape)}"
        )
    if use_packed_ue8m0 and x_sf.dtype != torch.int32:
        raise ValueError("packed UE8M0 scale factors are represented as int32")

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    if use_tma_aligned_col_major_sf:
        aligned_tokens = _align(pos_to_expert.numel(), 4)
        out_sf = torch.zeros(
            (x_sf.shape[1], aligned_tokens),
            dtype=x_sf.dtype,
            device=x_sf.device,
        )
        # Intentionally return a transposed view to emulate the TMA-aligned
        # column-major scale-factor layout while preserving public shape [X, S].
        out_sf = out_sf[:, : pos_to_expert.numel()].T
    else:
        out_sf = torch.zeros(
            (pos_to_expert.numel(), x_sf.shape[1]),
            dtype=x_sf.dtype,
            device=x_sf.device,
        )

    valid = token_topk_to_pos >= 0
    if valid.any():
        src_rows = _source_rows(token_topk_to_pos)
        out_sf[token_topk_to_pos[valid].long()] = x_sf[src_rows[valid].long()]
    return out, out_sf

def _make_expert_major_mapping(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic expert-major padded mapping for test inputs.

    Only the observable mapping contract required by expand_to_fused tests is
    reproduced here. This is not intended to match every implementation detail
    of moe_get_fused_mapping.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be 2-D, got shape={tuple(topk_ids.shape)}")

    counts = [0] * num_experts
    for eid in topk_ids.flatten().tolist():
        if eid < 0:
            continue
        if eid >= num_experts:
            raise ValueError(f"expert id {eid} is outside [0, {num_experts})")
        counts[eid] += 1

    offsets = [0] * (num_experts + 1)
    for eid, count in enumerate(counts):
        offsets[eid + 1] = offsets[eid] + _align(count, block_size)

    token_topk_to_pos = torch.full_like(topk_ids, -1, dtype=torch.int32)
    pos_to_expert = torch.full((offsets[-1],), -1, dtype=torch.int32, device=topk_ids.device)

    write_ptr = offsets[:-1].copy()
    for token in range(topk_ids.shape[0]):
        for kth in range(topk_ids.shape[1]):
            eid = int(topk_ids[token, kth].item())
            if eid < 0:
                continue
            pos = write_ptr[eid]
            token_topk_to_pos[token, kth] = pos
            pos_to_expert[pos] = eid
            write_ptr[eid] += 1

    return token_topk_to_pos, pos_to_expert

def _make_manifest_mapping(
    num_tokens: int,
    topk: int,
    num_expanded_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a shape-driven mapping matching manifest workload dimensions."""
    token_topk_to_pos = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    pos_to_expert = torch.full((num_expanded_tokens,), -1, dtype=torch.int32)

    num_valid = min(num_tokens * topk, num_expanded_tokens)
    if num_valid == 0:
        return token_topk_to_pos, pos_to_expert

    flat = token_topk_to_pos.view(-1)
    flat[:num_valid] = torch.arange(num_valid, dtype=torch.int32)
    pos_to_expert[:num_valid] = torch.arange(num_valid, dtype=torch.int32) % 16
    return token_topk_to_pos, pos_to_expert

def _make_topk_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    invalid_every: int | None = None,
) -> torch.Tensor:
    token = torch.arange(num_tokens, dtype=torch.int32).view(-1, 1)
    kth = torch.arange(topk, dtype=torch.int32).view(1, -1)
    topk_ids = (token * 3 + kth * 5) % num_experts
    if invalid_every is not None and num_tokens > 0:
        topk_ids[::invalid_every, -1] = -1
    return topk_ids

def _assert_expand_obeys_mapping(
    out: torch.Tensor,
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> None:
    assert out.shape == (pos_to_expert.numel(), x.shape[1])
    assert out.dtype == x.dtype

    valid = token_topk_to_pos >= 0
    targeted = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    if valid.any():
        target_pos = token_topk_to_pos[valid].long()
        src_rows = _source_rows(token_topk_to_pos)[valid].long()
        targeted[target_pos] = True
        assert torch.unique(target_pos).numel() == target_pos.numel()
        if _is_float8_dtype(out.dtype):
            assert torch.equal(
                out.view(torch.uint8)[target_pos],
                x.view(torch.uint8)[src_rows],
            )
        else:
            _assert_exact(out[target_pos], x[src_rows])
    if _is_float8_dtype(out.dtype):
        out_bytes = out.view(torch.uint8)
        assert torch.equal(out_bytes[~targeted], torch.zeros_like(out_bytes[~targeted]))
    else:
        assert torch.equal(out[~targeted], torch.zeros_like(out[~targeted]))

def _assert_sf_obeys_mapping(
    out_sf: torch.Tensor,
    x_sf: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> None:
    assert out_sf.shape == (pos_to_expert.numel(), x_sf.shape[1])
    assert out_sf.dtype == x_sf.dtype

    valid = token_topk_to_pos >= 0
    targeted = torch.zeros(out_sf.shape[0], dtype=torch.bool, device=out_sf.device)
    if valid.any():
        target_pos = token_topk_to_pos[valid].long()
        src_rows = _source_rows(token_topk_to_pos)
        targeted[target_pos] = True
        assert torch.unique(target_pos).numel() == target_pos.numel()
        _assert_exact(out_sf[target_pos], x_sf[src_rows[valid].long()])
    assert torch.equal(out_sf[~targeted], torch.zeros_like(out_sf[~targeted]))

def _run_reference_case(
    *,
    num_tokens: int,
    hidden: int,
    topk: int,
    num_experts: int,
    block_size: int,
    dtype: torch.dtype = torch.bfloat16,
    invalid_every: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_ids = _make_topk_ids(num_tokens, topk, num_experts, invalid_every=invalid_every)
    token_topk_to_pos, pos_to_expert = _make_expert_major_mapping(
        topk_ids,
        num_experts=num_experts,
        block_size=block_size,
    )
    x = _make_x(num_tokens, hidden, dtype)
    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert out.shape == (pos_to_expert.numel(), hidden)
    assert out.dtype == dtype
    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)
    return out, x, token_topk_to_pos, pos_to_expert

@pytest.mark.smoke
def test_moe_expand_to_fused_manual_mapping() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ],
        dtype=torch.bfloat16,
    )
    token_topk_to_pos = torch.tensor(
        [
            [0, 3],
            [1, 4],
            [2, -1],
        ],
        dtype=torch.int32,
    )
    pos_to_expert = torch.tensor([0, 0, 0, 1, 1, -1, -1], dtype=torch.int32)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    _assert_exact(out, expected)

@pytest.mark.smoke
def test_moe_expand_to_fused_invalid_topk_and_padding() -> None:
    _, x, token_topk_to_pos, pos_to_expert = _run_reference_case(
        num_tokens=8,
        hidden=16,
        topk=4,
        num_experts=4,
        block_size=8,
        dtype=torch.float16,
        invalid_every=2,
    )
    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert (token_topk_to_pos < 0).any()
    assert (pos_to_expert < 0).any()
    assert torch.equal(out[pos_to_expert < 0], torch.zeros_like(out[pos_to_expert < 0]))

@pytest.mark.smoke
def test_moe_expand_to_fused_does_not_weight_or_reduce() -> None:
    x = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float16)
    token_topk_to_pos = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    pos_to_expert = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    _assert_exact(out[0], x[0])
    _assert_exact(out[2], x[0])
    assert not torch.equal(out[0], x[0] * 2)

@pytest.mark.smoke
def test_moe_expand_to_fused_random_mapping() -> None:
    torch.manual_seed(0)
    num_tokens, topk, num_experts, hidden = 13, 6, 8, 37
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32)
    topk_ids[::4, -1] = -1
    token_topk_to_pos, pos_to_expert = _make_expert_major_mapping(
        topk_ids,
        num_experts=num_experts,
        block_size=16,
    )
    x = _make_x(num_tokens, hidden, torch.bfloat16)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)

@pytest.mark.parametrize(
    "num_expanded_tokens",
    [
        pytest.param(0, marks=pytest.mark.smoke, id="no-expanded-slots"),
        pytest.param(4, marks=pytest.mark.full, id="padding-only-slots"),
    ],
)
def test_moe_expand_to_fused_empty_tokens_reference_only(num_expanded_tokens: int) -> None:
    x = torch.empty((0, 576), dtype=torch.bfloat16)
    token_topk_to_pos = torch.empty((0, 8), dtype=torch.int32)
    pos_to_expert = torch.full((num_expanded_tokens,), -1, dtype=torch.int32)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert out.shape == (num_expanded_tokens, 576)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros_like(out))

@pytest.mark.smoke
def test_moe_expand_to_fused_all_invalid_mapping() -> None:
    x = _make_x(num_tokens=3, hidden=16, dtype=torch.bfloat16)
    token_topk_to_pos = torch.full((3, 4), -1, dtype=torch.int32)
    pos_to_expert = torch.full((8,), -1, dtype=torch.int32)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert not (token_topk_to_pos >= 0).any()
    assert torch.equal(out, torch.zeros_like(out))

@pytest.mark.parametrize(
    "hidden",
    [
        pytest.param(
            hidden,
            marks=pytest.mark.smoke
            if hidden == _DOCUMENTED_HIDDEN_SIZES[0]
            else pytest.mark.full,
            id=f"h{hidden}",
        )
        for hidden in _DOCUMENTED_HIDDEN_SIZES
    ],
)
def test_moe_expand_to_fused_documented_hidden_sizes(hidden: int) -> None:
    _run_reference_case(
        num_tokens=5,
        hidden=hidden,
        topk=2,
        num_experts=4,
        block_size=4,
    )

@pytest.mark.parametrize(
    "topk",
    [
        pytest.param(
            topk,
            marks=pytest.mark.smoke if topk == _DOCUMENTED_TOPK[0] else pytest.mark.full,
            id=f"topk{topk}",
        )
        for topk in _DOCUMENTED_TOPK
    ],
)
def test_moe_expand_to_fused_documented_topk(topk: int) -> None:
    _run_reference_case(
        num_tokens=7,
        hidden=128,
        topk=topk,
        num_experts=12,
        block_size=8,
    )

@pytest.mark.smoke
def test_moe_expand_to_fused_manifest_decode_padding_shape() -> None:
    x = _make_x(num_tokens=1, hidden=64, dtype=torch.bfloat16)
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens=1,
        topk=8,
        num_expanded_tokens=16,
    )

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert token_topk_to_pos.numel() == 8
    assert pos_to_expert.numel() == 16
    assert out.shape == (16, 64)
    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)

@pytest.mark.smoke
def test_moe_expand_to_fused_num_send_tokens_4001_boundary() -> None:
    _run_reference_case(
        num_tokens=4001,
        hidden=1,
        topk=8,
        num_experts=16,
        block_size=64,
        invalid_every=17,
    )

@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=pytest.mark.smoke, id="bf16"),
        pytest.param(torch.float16, marks=pytest.mark.smoke, id="fp16"),
    ],
)
def test_moe_expand_to_fused_supported_dtypes(dtype: torch.dtype) -> None:
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens=5,
        topk=2,
        num_expanded_tokens=10,
    )
    x = _make_x(5, 7, dtype)

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)

@pytest.mark.parametrize(
    ("label", "num_tokens", "hidden", "topk", "num_expanded_tokens"),
    _BASE_MANIFEST_WORKLOADS,
)
def test_moe_expand_to_fused_manifest_workload_shapes(
    label: str,
    num_tokens: int,
    hidden: int,
    topk: int,
    num_expanded_tokens: int,
) -> None:
    assert label
    x = _make_x(num_tokens, hidden, torch.bfloat16)
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens,
        topk,
        num_expanded_tokens,
    )

    out = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    assert out.shape == (num_expanded_tokens, hidden)
    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)

@pytest.mark.skipif(_FP8_E4M3 is None, reason="torch fp8 is unavailable")
@pytest.mark.parametrize(
    "num_per_channels",
    [
        pytest.param(32, marks=pytest.mark.smoke, id="npc32"),
        pytest.param(128, marks=pytest.mark.smoke, id="npc128"),
    ],
)
@pytest.mark.parametrize(
    ("use_tma_aligned_col_major_sf", "sf_dtype", "use_packed_ue8m0"),
    [
        pytest.param(False, torch.float32, False, id="row-major-fp32-sf"),
        pytest.param(True, torch.int32, True, id="tma-col-major-packed-ue8m0"),
    ],
)
def test_moe_expand_to_fused_with_sf(
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    sf_dtype: torch.dtype,
    use_packed_ue8m0: bool,
) -> None:
    num_tokens, topk, num_expanded_tokens, hidden = 6, 8, 64, 576
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens,
        topk,
        num_expanded_tokens,
    )
    x = _make_x(num_tokens, hidden, _FP8_E4M3)
    channels = _sf_channels(hidden, num_per_channels, packed=use_packed_ue8m0)
    x_sf = _make_scale_factors(num_tokens, channels, sf_dtype)

    out, out_sf = _ref_moe_expand_to_fused_with_sf(
        x,
        x_sf,
        num_per_channels,
        token_topk_to_pos,
        pos_to_expert,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)
    _assert_sf_obeys_mapping(out_sf, x_sf, token_topk_to_pos, pos_to_expert)
    assert torch.equal(out_sf[pos_to_expert < 0], torch.zeros_like(out_sf[pos_to_expert < 0]))
    if use_tma_aligned_col_major_sf and channels > 1 and pos_to_expert.numel() > 0:
        assert out_sf.stride() == (1, _align(pos_to_expert.numel(), 4))
    else:
        assert out_sf.is_contiguous()

@pytest.mark.skipif(_FP8_E4M3 is None, reason="torch fp8 is unavailable")
@pytest.mark.parametrize(
    (
        "label",
        "num_tokens",
        "hidden",
        "topk",
        "num_expanded_tokens",
        "num_per_channels",
        "use_tma_aligned_col_major_sf",
        "sf_dtype",
        "use_packed_ue8m0",
    ),
    _SF_MANIFEST_WORKLOADS,
)
def test_moe_expand_to_fused_with_sf_manifest_workload_shapes(
    label: str,
    num_tokens: int,
    hidden: int,
    topk: int,
    num_expanded_tokens: int,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    sf_dtype: torch.dtype,
    use_packed_ue8m0: bool,
) -> None:
    assert label
    x = _make_x(num_tokens, hidden, _FP8_E4M3)
    channels = _sf_channels(hidden, num_per_channels, packed=use_packed_ue8m0)
    x_sf = _make_scale_factors(num_tokens, channels, sf_dtype)
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens,
        topk,
        num_expanded_tokens,
    )

    out, out_sf = _ref_moe_expand_to_fused_with_sf(
        x,
        x_sf,
        num_per_channels,
        token_topk_to_pos,
        pos_to_expert,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    assert out.shape == (num_expanded_tokens, hidden)
    assert out_sf.shape == (num_expanded_tokens, channels)
    _assert_expand_obeys_mapping(out, x, token_topk_to_pos, pos_to_expert)
    _assert_sf_obeys_mapping(out_sf, x_sf, token_topk_to_pos, pos_to_expert)

@pytest.mark.skipif(_FP8_E4M3 is None, reason="torch fp8 is unavailable")
@pytest.mark.smoke
def test_moe_expand_to_fused_with_sf_reference_rejects_shape_rules() -> None:
    x = _make_x(num_tokens=2, hidden=576, dtype=_FP8_E4M3)
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens=2,
        topk=2,
        num_expanded_tokens=4,
    )

    x_sf_row = _make_scale_factors(2, _sf_channels(576, 128, packed=False), torch.float32)
    with pytest.raises(ValueError, match="unsupported num_per_channels"):
        _ref_moe_expand_to_fused_with_sf(
            x,
            x_sf_row,
            64,
            token_topk_to_pos,
            pos_to_expert,
        )

    x_sf_packed = _make_scale_factors(2, _sf_channels(576, 128, packed=True), torch.int32)
    with pytest.raises(ValueError, match="packed UE8M0 scale factors require"):
        _ref_moe_expand_to_fused_with_sf(
            x,
            x_sf_packed,
            128,
            token_topk_to_pos,
            pos_to_expert,
            use_packed_ue8m0=True,
        )

    bad_shape = _make_scale_factors(2, x_sf_row.shape[1] + 1, torch.float32)
    with pytest.raises(ValueError, match="x_sf must have shape"):
        _ref_moe_expand_to_fused_with_sf(
            x,
            bad_shape,
            128,
            token_topk_to_pos,
            pos_to_expert,
        )

    bad_packed_dtype = _make_scale_factors(
        2,
        _sf_channels(576, 128, packed=True),
        torch.float32,
    )
    with pytest.raises(ValueError, match="packed UE8M0 scale factors are represented"):
        _ref_moe_expand_to_fused_with_sf(
            x,
            bad_packed_dtype,
            128,
            token_topk_to_pos,
            pos_to_expert,
            use_tma_aligned_col_major_sf=True,
            use_packed_ue8m0=True,
        )

@pytest.mark.smoke
def test_moe_expand_to_fused_reference_rejects_duplicate_positions() -> None:
    x = torch.randn(2, 4, dtype=torch.float16)
    token_topk_to_pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    pos_to_expert = torch.tensor([0, 0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="duplicate valid positions"):
        _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

@pytest.mark.smoke
def test_moe_expand_to_fused_reference_rejects_padding_target() -> None:
    x = torch.randn(1, 4, dtype=torch.float16)
    token_topk_to_pos = torch.tensor([[0]], dtype=torch.int32)
    pos_to_expert = torch.tensor([-1], dtype=torch.int32)

    with pytest.raises(ValueError, match="padding position"):
        _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

@pytest.mark.smoke
def test_moe_expand_to_fused_reference_rejects_manifest_dtype_mismatch() -> None:
    x = torch.randn(1, 4, dtype=torch.float16)
    token_topk_to_pos = torch.tensor([[0]], dtype=torch.int64)
    pos_to_expert = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="token_topk_to_pos must be int32"):
        _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.smoke
def test_moe_expand_to_fused_kernel_matches_reference_smoke() -> None:
    pytest.importorskip("tilelang")
    module = pytest.importorskip("tileops.kernels.moe.expand_to_fused")
    kernel_cls = getattr(module, "MoeExpandToFusedKernel")

    x = _make_x(num_tokens=4, hidden=64, dtype=torch.bfloat16).cuda()
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens=4,
        topk=2,
        num_expanded_tokens=12,
    )
    token_topk_to_pos = token_topk_to_pos.cuda()
    pos_to_expert = pos_to_expert.cuda()

    ref = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    kernel = kernel_cls(
        hidden=x.shape[1],
        num_topk=token_topk_to_pos.shape[1],
        dtype=x.dtype,
    )
    out = kernel(x, token_topk_to_pos, pos_to_expert)
    torch.cuda.synchronize()

    _assert_exact(out, ref)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.smoke
def test_moe_expand_to_fused_op_matches_reference_smoke() -> None:
    module = pytest.importorskip("tileops.ops.moe.expand_to_fused")
    op_cls = getattr(module, "MoeExpandToFusedFwdOp")

    x = _make_x(num_tokens=4, hidden=64, dtype=torch.bfloat16).cuda()
    token_topk_to_pos, pos_to_expert = _make_manifest_mapping(
        num_tokens=4,
        topk=2,
        num_expanded_tokens=12,
    )
    token_topk_to_pos = token_topk_to_pos.cuda()
    pos_to_expert = pos_to_expert.cuda()

    ref = _ref_moe_expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    out = op_cls()(x, token_topk_to_pos, pos_to_expert)

    _assert_exact(out, ref)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
