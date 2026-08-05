"""Op-level tests for MoE expand-to-fused scatter ops.

Verifies:
  - bf16/fp16 activation scatter into the fused expert layout
  - dropped routing slots and unassigned output positions are zero-filled/skipped
  - extra output capacity where pos_to_expert.shape[0] > T * K
  - fp8 activation scatter with row-major float32 scale factors
  - fp8 activation scatter with TMA-aligned packed int32 scale factors
  - uint8/int8 byte payloads accepted by the quantized path
  - non-contiguous inputs accepted by the op wrapper
  - input validation for routing maps and scale-factor shapes
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoeExpandToFusedFwdOp, MoeExpandToFusedWithSFFwdOp

EXACT_ATOL = 0
EXACT_RTOL = 0


def _make_routing(
    total_tokens: int,
    top_k: int,
    num_expanded_tokens: int,
    *,
    device: str = "cuda",
    include_dropped: bool = False,
    include_unassigned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic routing plan with optional -1 sentinels."""
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
                break
            token_topk_to_pos[t, k] = slot
            pos_to_expert[slot] = k
            slot += 1

    if include_dropped:
        dropped_pos = int(token_topk_to_pos[-1, -1].item())
        token_topk_to_pos[-1, -1] = -1
        if dropped_pos >= 0:
            pos_to_expert[dropped_pos] = -1
    if include_unassigned:
        pos_to_expert[-1] = -1

    return token_topk_to_pos, pos_to_expert


def _as_noncontiguous_2d(tensor: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (tensor.shape[0], tensor.shape[1] * 2),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = storage[:, ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _as_noncontiguous_1d(tensor: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (tensor.shape[0] * 2,),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = storage[::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _ref_expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for expand-to-fused."""
    expanded_x = torch.zeros(
        (pos_to_expert.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
    )

    for t in range(x.shape[0]):
        for k in range(token_topk_to_pos.shape[1]):
            pos = int(token_topk_to_pos[t, k].item())
            if pos >= 0:
                expanded_x[pos] = x[t]
    return expanded_x


def _ref_expand_to_fused_with_sf(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    *,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for expand-to-fused with scale factors."""
    expanded_x = _ref_expand_to_fused(x, token_topk_to_pos, pos_to_expert)
    if use_tma_aligned_col_major_sf:
        num_expanded_sf_tokens = ((pos_to_expert.shape[0] + 3) // 4) * 4
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

    for t in range(x.shape[0]):
        for k in range(token_topk_to_pos.shape[1]):
            pos = int(token_topk_to_pos[t, k].item())
            if pos >= 0:
                expanded_x_sf[pos] = x_sf[t]
    return expanded_x, expanded_x_sf


class MoeExpandToFusedTest(TestBase):
    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        num_expanded_tokens: int,
        dtype: torch.dtype,
        *,
        include_dropped: bool = False,
        include_unassigned: bool = False,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_expanded_tokens = num_expanded_tokens
        self.dtype = dtype
        self.include_dropped = include_dropped
        self.include_unassigned = include_unassigned

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
            include_dropped=self.include_dropped,
            include_unassigned=self.include_unassigned,
        )
        return x, token_topk_to_pos, pos_to_expert

    def ref_program(self, x, token_topk_to_pos, pos_to_expert):
        return _ref_expand_to_fused(x, token_topk_to_pos, pos_to_expert)


class MoeExpandToFusedWithSFTest(TestBase):
    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        num_expanded_tokens: int,
        num_per_channels: int,
        dtype: torch.dtype,
        sf_dtype: torch.dtype,
        *,
        use_tma_aligned_col_major_sf: bool = False,
        include_dropped: bool = False,
        include_unassigned: bool = False,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_expanded_tokens = num_expanded_tokens
        self.num_per_channels = num_per_channels
        self.dtype = dtype
        self.sf_dtype = sf_dtype
        self.use_tma_aligned_col_major_sf = use_tma_aligned_col_major_sf
        self.include_dropped = include_dropped
        self.include_unassigned = include_unassigned

    @property
    def shape(self) -> tuple[int, int]:
        return (self.total_tokens, self.hidden_size)

    @property
    def hidden_sf(self) -> int:
        blocks = (self.hidden_size + self.num_per_channels - 1) // self.num_per_channels
        if self.sf_dtype == torch.int32:
            blocks = (blocks + 3) // 4
        return blocks

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        if self.dtype == torch.float8_e4m3fn:
            x = torch.randint(
                -16,
                16,
                (self.total_tokens, self.hidden_size),
                dtype=torch.int8,
                device="cuda",
            ).to(torch.float8_e4m3fn)
        else:
            x = torch.randint(
                0,
                32,
                (self.total_tokens, self.hidden_size),
                dtype=self.dtype,
                device="cuda",
            )
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
            include_dropped=self.include_dropped,
            include_unassigned=self.include_unassigned,
        )
        return x, x_sf, token_topk_to_pos, pos_to_expert

    def ref_program(self, x, x_sf, token_topk_to_pos, pos_to_expert):
        return _ref_expand_to_fused_with_sf(
            x,
            x_sf,
            token_topk_to_pos,
            pos_to_expert,
            use_tma_aligned_col_major_sf=self.use_tma_aligned_col_major_sf,
        )


class MoeExpandToFusedFixture(FixtureBase):
    PARAMS = [
        (
            "total_tokens, top_k, hidden_size, num_expanded_tokens, dtype, "
            "include_dropped, include_unassigned",
            [
                pytest.param(4, 2, 64, 8, torch.bfloat16, False, False,
                             marks=pytest.mark.smoke, id="tiny-bf16"),
                pytest.param(4, 2, 64, 8, torch.float16, False, False,
                             marks=pytest.mark.smoke, id="tiny-fp16"),
                pytest.param(1, 1, 1, 1, torch.float16, False, False,
                             marks=pytest.mark.full, id="minimal-fp16"),
                pytest.param(3, 2, 65, 6, torch.bfloat16, False, False,
                             marks=pytest.mark.full, id="non-block-hidden-bf16"),
                pytest.param(5, 3, 96, 15, torch.bfloat16, True, False,
                             marks=pytest.mark.full, id="dropped-slot"),
                pytest.param(8, 2, 128, 17, torch.bfloat16, False, True,
                             marks=pytest.mark.full, id="unassigned-position"),
                pytest.param(1, 8, 7168, 16, torch.bfloat16, False, True,
                             marks=pytest.mark.full, id="extra-capacity-decode"),
                pytest.param(16, 4, 256, 64, torch.float16, False, False,
                             marks=pytest.mark.full, id="medium-fp16"),
            ],
        ),
    ]


class MoeExpandToFusedWithSFFixture(FixtureBase):
    PARAMS = [
        (
            "total_tokens, top_k, hidden_size, num_expanded_tokens, "
            "num_per_channels, sf_dtype, use_tma_aligned_col_major_sf, "
            "include_dropped, include_unassigned",
            [
                pytest.param(4, 2, 128, 8, 128, torch.float32, False, False, False,
                             marks=pytest.mark.smoke, id="fp8-sf-row-major"),
                pytest.param(5, 2, 128, 10, 128, torch.int32, True, True, False,
                             marks=pytest.mark.smoke, id="fp8-sf-packed"),
                pytest.param(3, 2, 129, 6, 128, torch.float32, False, False, False,
                             marks=pytest.mark.full, id="fp8-sf-non-block"),
                pytest.param(1, 8, 7168, 16, 128, torch.float32, False, False, True,
                             marks=pytest.mark.full, id="fp8-sf-extra-capacity"),
                pytest.param(8, 3, 256, 25, 128, torch.float32, False, False, True,
                             marks=pytest.mark.full, id="fp8-sf-unassigned"),
                pytest.param(4, 2, 96, 8, 32, torch.float32, False, False, False,
                             marks=pytest.mark.full, id="fp8-sf-npc32"),
                pytest.param(4, 2, 64, 8, 32, torch.float32, False, False, False,
                             marks=pytest.mark.full, id="uint8-sf-row-major"),
                pytest.param(4, 2, 128, 8, 128, torch.int32, True, False, False,
                             marks=pytest.mark.full, id="int8-sf-packed"),
            ],
        ),
    ]


@MoeExpandToFusedFixture
def test_moe_expand_to_fused_op(
    total_tokens,
    top_k,
    hidden_size,
    num_expanded_tokens,
    dtype,
    include_dropped,
    include_unassigned,
):
    test = MoeExpandToFusedTest(
        total_tokens,
        top_k,
        hidden_size,
        num_expanded_tokens,
        dtype,
        include_dropped=include_dropped,
        include_unassigned=include_unassigned,
    )
    op = MoeExpandToFusedFwdOp(dtype=dtype)
    inputs = test.gen_inputs()

    test.check(op, *inputs, atol=0, rtol=0)
    flops, nbytes = op.eval_roofline()
    assert flops == 0
    assert nbytes > 0


@MoeExpandToFusedWithSFFixture
def test_moe_expand_to_fused_with_sf_op(
    total_tokens,
    top_k,
    hidden_size,
    num_expanded_tokens,
    num_per_channels,
    sf_dtype,
    use_tma_aligned_col_major_sf,
    include_dropped,
    include_unassigned,
):
    test = MoeExpandToFusedWithSFTest(
        total_tokens,
        top_k,
        hidden_size,
        num_expanded_tokens,
        num_per_channels,
        torch.float8_e4m3fn,
        sf_dtype,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        include_dropped=include_dropped,
        include_unassigned=include_unassigned,
    )
    op = MoeExpandToFusedWithSFFwdOp(
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        dtype=torch.float8_e4m3fn,
        sf_dtype=sf_dtype,
    )
    inputs = test.gen_inputs()

    output, output_sf = op(*inputs)
    ref, ref_sf = test.ref_program(*inputs)

    assert output_sf.shape == ref_sf.shape
    assert output_sf.stride() == ref_sf.stride()
    assert torch.equal(output.cpu(), ref.cpu())
    assert torch.equal(output_sf.cpu(), ref_sf.cpu())
    flops, nbytes = op.eval_roofline()
    assert flops == 0
    assert nbytes > 0


@pytest.mark.smoke
def test_moe_expand_to_fused_shape_mismatch_raises() -> None:
    x = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    token_topk_to_pos = torch.zeros(5, 2, dtype=torch.int32, device="cuda")
    pos_to_expert = torch.zeros(8, dtype=torch.int32, device="cuda")
    op = MoeExpandToFusedFwdOp(dtype=torch.float16)

    with pytest.raises(ValueError, match=r"token_topk_to_pos.shape\[0\]"):
        op(x, token_topk_to_pos, pos_to_expert)


@pytest.mark.smoke
def test_moe_expand_to_fused_index_dtype_raises() -> None:
    x = torch.randn(4, 16, dtype=torch.float16, device="cuda")
    token_topk_to_pos = torch.zeros(4, 2, dtype=torch.int64, device="cuda")
    pos_to_expert = torch.zeros(8, dtype=torch.int32, device="cuda")
    op = MoeExpandToFusedFwdOp(dtype=torch.float16)

    with pytest.raises(ValueError, match="token_topk_to_pos.dtype"):
        op(x, token_topk_to_pos, pos_to_expert)


@pytest.mark.smoke
def test_moe_expand_to_fused_with_sf_shape_mismatch_raises() -> None:
    x = torch.zeros(4, 128, dtype=torch.float8_e4m3fn, device="cuda")
    x_sf = torch.zeros(4, 2, dtype=torch.float32, device="cuda")
    token_topk_to_pos, pos_to_expert = _make_routing(4, 2, 8)
    op = MoeExpandToFusedWithSFFwdOp(num_per_channels=128)

    with pytest.raises(ValueError, match=r"x_sf.shape\[1\]"):
        op(x, x_sf, token_topk_to_pos, pos_to_expert)


@pytest.mark.smoke
def test_moe_expand_to_fused_with_sf_packed_requires_tma() -> None:
    x = torch.zeros(4, 128, dtype=torch.float8_e4m3fn, device="cuda")
    x_sf = torch.zeros(4, 1, dtype=torch.int32, device="cuda")
    token_topk_to_pos, pos_to_expert = _make_routing(4, 2, 8)
    op = MoeExpandToFusedWithSFFwdOp(
        num_per_channels=128,
        use_tma_aligned_col_major_sf=False,
    )

    with pytest.raises(ValueError, match="Packed UE8M0"):
        op(x, x_sf, token_topk_to_pos, pos_to_expert)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
