"""MoE expand-to-fused kernels: scatter token activations into the fused layout.

Each token row of ``x`` is broadcast to the ``num_topk`` expanded positions the
routing plan assigned to it, producing the row-contiguous, expert-major input
the fused expert GEMM consumes. Positions are numbered so that all rows of one
expert are adjacent, so the plain scatter below already lands the data in
expert-major order -- the kernel never sorts, it only follows the plan.

One block per position (``max(num_tokens, num_expanded_tokens)`` blocks):
  - Blocks below ``num_expanded_tokens`` zero-fill their output row when the
    position is unassigned (``pos_to_expert[p] < 0``).
  - Blocks below ``num_tokens`` load their token row into a fragment and
    scatter it to every non-negative position in ``token_topk_to_pos[t]``.
    A negative entry marks a dropped token-expert slot and is skipped.

``num_tokens`` and ``num_expanded_tokens`` are TileLang dynamic symbols, so one
compiled kernel serves every batch size at a given static configuration.

Quantized activations
---------------------
The same program carries a per-block scale factor (SF) tensor alongside the
activations, moved by the identical scatter. ``num_per_channels`` (32 or 128)
sets the block width, giving ``hidden_sf = ceil(hidden / num_per_channels)``
scale factors per row. Two SF encodings are supported:

  - **float32**, row-major ``[num_expanded_tokens, hidden_sf]``.
  - **Packed UE8M0 as int32**: four UE8M0 exponent bytes packed per int32 word,
    so ``hidden_sf`` shrinks by a further factor of 4. Always paired with the
    TMA-aligned column-major layout.

With ``use_tma_aligned_col_major_sf`` the SF output is written transposed as
``[hidden_sf, num_expanded_tokens]`` over a token dimension padded to a
multiple of 4, which is the layout the downstream TMA-based GEMM expects. The
kernel takes it as a ``T.StridedTensor`` with a runtime leading stride, so the
caller can hand it a slice of the padded allocation.

Because the activation payload is only ever copied, never interpreted, the
element dtype is opaque to the kernel: fp8 (``float8_e4m3fn``) and 2-way-packed
fp4 (carried in a ``uint8``/``int8`` buffer of ``hidden / 2`` columns) both work
through the same code path as bf16/fp16.

Inputs:
  x                 [num_tokens, hidden]           activations
  x_sf              [num_tokens, hidden_sf]        scale factors (SF path only)
  token_topk_to_pos [num_tokens, num_topk]         int32 (token, slot) -> position
  pos_to_expert     [num_expanded_tokens]          int32 position -> expert (-1 = unassigned)

Outputs:
  expanded_x        [num_expanded_tokens, hidden]  same dtype as x
  expanded_x_sf     scale factors in the requested layout (SF path only)
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MoeExpandToFusedKernel", "MoeExpandToFusedWithSFKernel"]

_DEFAULT_NUM_THREADS = 256
_SMALL_HIDDEN_PREFILL_THREADS = 128
_SMALL_HIDDEN_THRESHOLD = 4096
_PREFILL_TOKEN_THRESHOLD = 2048

# Scale-factor block widths the downstream quantized GEMM understands.
_SUPPORTED_NUM_PER_CHANNELS = (32, 128)

# UE8M0 exponent bytes packed per int32 word.
_UE8M0_PER_INT32 = 4

# Token-dimension multiple the TMA-aligned column-major SF layout pads to.
_SF_TOKEN_ALIGN = 4


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _hidden_sf(hidden: int, num_per_channels: int, use_packed_ue8m0: bool) -> int:
    """Return the scale-factor column count for one row of ``hidden`` elements."""
    hidden_sf = _ceil_div(hidden, num_per_channels)
    if use_packed_ue8m0:
        hidden_sf = _ceil_div(hidden_sf, _UE8M0_PER_INT32)
    return hidden_sf


def select_expand_to_fused_num_threads(
    hidden: int,
    num_tokens: int,
    has_scale_factors: bool,
) -> int:
    """Choose the copy/scatter block width for the observed workload shape.

    MetaX C500 profiling showed that 256 threads improves the long-row copy
    path, while the 3072-wide non-SF prefill case has enough blocks that 128
    threads gives better occupancy.
    """
    if (
        not has_scale_factors
        and hidden <= _SMALL_HIDDEN_THRESHOLD
        and num_tokens >= _PREFILL_TOKEN_THRESHOLD
    ):
        return _SMALL_HIDDEN_PREFILL_THREADS
    return _DEFAULT_NUM_THREADS


@functools.lru_cache(maxsize=32)
def _expand_to_fused_kernel(
    hidden: int,
    num_topk: int,
    num_per_channels: Optional[int],
    use_tma_aligned_col_major_sf: Optional[bool],
    use_packed_ue8m0: Optional[bool],
    x_dtype: str,
    sf_dtype: str,
    num_threads: int,
):
    """Build the expand-to-fused prim_func for one static configuration.

    Args:
        hidden: Hidden dimension H, in elements of ``x_dtype``.
        num_topk: Number of expert slots per token K.
        num_per_channels: Channels per scale-factor block (32 or 128), or None
            to build the unquantized path that carries no scale factors.
        use_tma_aligned_col_major_sf: Whether the SF output is transposed to
            ``[hidden_sf, num_expanded_tokens]``. Ignored when
            ``num_per_channels`` is None.
        use_packed_ue8m0: Whether scale factors are UE8M0 bytes packed four to
            an int32. Ignored when ``num_per_channels`` is None.
        x_dtype: TileLang dtype string for the activations.
        sf_dtype: TileLang dtype string for the scale factors. Ignored when
            ``num_per_channels`` is None; pass ``x_dtype``.

    Returns:
        A ``@tilelang.jit`` builder returning the compiled kernel.
    """
    # Round the element loop up to a whole number of threads so every thread
    # takes the same trip count; the tail iterations index past `hidden` and
    # are predicated out against the declared buffer extent.
    hidden_aligned = _align(hidden, num_threads)

    if num_per_channels is not None:
        hidden_sf = _hidden_sf(hidden, num_per_channels, bool(use_packed_ue8m0))
        hidden_sf_aligned = _align(hidden_sf, num_threads)
    else:
        # The unquantized path still declares the SF parameters so both paths
        # share one program; the caller passes None and no SF code is emitted.
        hidden_sf, hidden_sf_aligned = 1, 1

    # Leading stride of the SF output. Runtime, not static: the caller may pass
    # a column slice of an allocation padded out to _SF_TOKEN_ALIGN.
    sf_stride = T.dynamic("sf_stride")
    num_tokens = T.dynamic("num_tokens")
    num_expanded_tokens = T.dynamic("num_expanded_tokens")
    num_blocks = T.max(num_tokens, num_expanded_tokens)

    sf_shape = (
        (hidden_sf, num_expanded_tokens)
        if use_tma_aligned_col_major_sf
        else (num_expanded_tokens, hidden_sf)
    )

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def _expand_to_fused():

        @T.prim_func
        def _expand_to_fused_main(
            x: T.Tensor((num_tokens, hidden), x_dtype),
            x_sf: T.Tensor((num_tokens, hidden_sf), sf_dtype),
            expanded_x: T.Tensor((num_expanded_tokens, hidden), x_dtype),
            expanded_x_sf: T.StridedTensor(sf_shape, (sf_stride, 1), sf_dtype),
            token_topk_to_pos: T.Tensor((num_tokens, num_topk), "int32"),
            pos_to_expert: T.Tensor((num_expanded_tokens,), "int32"),
        ):
            with T.Kernel(num_blocks, threads=num_threads) as (pid_token,):
                pos_local = T.alloc_local((num_topk,), "int32")

                # Unassigned positions are never a scatter target, so they must
                # be zeroed here or the output row stays uninitialized.
                # Kept nested: TileLang lowers `and` to a non-short-circuiting
                # predicate, which would read pos_to_expert out of bounds on the
                # blocks that only exist to cover num_tokens.
                if pid_token < num_expanded_tokens:  # noqa: SIM102
                    if pos_to_expert[pid_token] < 0:
                        for i in T.Parallel(hidden_aligned):
                            expanded_x[pid_token, i] = 0
                        if num_per_channels is not None:
                            for i in T.Parallel(hidden_sf_aligned):
                                if use_tma_aligned_col_major_sf:
                                    expanded_x_sf[i, pid_token] = 0
                                else:
                                    expanded_x_sf[pid_token, i] = 0

                # The grid covers max(num_tokens, num_expanded_tokens); blocks
                # past the token count have no row to scatter.
                if pid_token >= num_tokens:
                    T.thread_return()
                T.assume(pid_token < num_tokens)

                x_fragment = T.alloc_fragment((hidden_aligned,), x_dtype)
                x_sf_fragment = T.alloc_fragment((hidden_sf_aligned,), sf_dtype)

                T.copy(token_topk_to_pos[pid_token, :], pos_local)
                T.copy(x[pid_token, :], x_fragment[0:hidden])
                if num_per_channels is not None:
                    T.copy(x_sf[pid_token, :], x_sf_fragment[0:hidden_sf])

                # One staged read, num_topk scattered writes: the row is
                # broadcast to every expert that selected this token.
                for k in T.serial(num_topk):
                    T.assume(pos_local[k] < num_expanded_tokens)
                    if pos_local[k] >= 0:
                        for i in T.Parallel(hidden_aligned):
                            expanded_x[pos_local[k], i] = x_fragment[i]
                        if num_per_channels is not None:
                            for i in T.Parallel(hidden_sf_aligned):
                                if use_tma_aligned_col_major_sf:
                                    expanded_x_sf[i, pos_local[k]] = x_sf_fragment[i]
                                else:
                                    expanded_x_sf[pos_local[k], i] = x_sf_fragment[i]

        return _expand_to_fused_main

    return _expand_to_fused


class MoeExpandToFusedKernel(Kernel):
    """Scatter token activations into the fused expert layout.

    Args:
        hidden: Hidden dimension H.
        num_topk: Number of expert slots per token K.
        dtype: Data type of the activations (bf16 or fp16).
        config: Optional config dict.

    Example:
        >>> kernel = MoeExpandToFusedKernel(hidden=128, num_topk=2)
        >>> expanded_x = kernel(x, token_topk_to_pos, pos_to_expert)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        dtype: torch.dtype = torch.bfloat16,
        num_threads: int = _DEFAULT_NUM_THREADS,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_topk = num_topk
        self.dtype = dtype
        self.num_threads = num_threads

        self._kernel_fn = _expand_to_fused_kernel(
            hidden,
            num_topk,
            None,
            None,
            None,
            self.dtype_str,
            self.dtype_str,
            num_threads,
        )

        self.init_config(config, tune=False)

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run expand-to-fused.

        Args:
            x: [num_tokens, hidden] token activations.
            token_topk_to_pos: [num_tokens, num_topk] int32 map from a
                (token, slot) pair to its expanded position; negative marks a
                dropped slot.
            pos_to_expert: [num_expanded_tokens] int32 map from an expanded
                position to its expert; negative marks an unassigned position.

        Returns:
            expanded_x: [num_expanded_tokens, hidden] activations in fused
                expert layout. Unassigned positions are zero-filled.
        """
        assert x.is_cuda and token_topk_to_pos.is_cuda and pos_to_expert.is_cuda
        assert x.is_contiguous() and token_topk_to_pos.is_contiguous()
        assert pos_to_expert.is_contiguous()
        assert token_topk_to_pos.dtype == torch.int32
        assert pos_to_expert.dtype == torch.int32
        assert x.shape[1] == self.hidden
        assert token_topk_to_pos.shape[1] == self.num_topk
        assert token_topk_to_pos.shape[0] == x.shape[0]

        num_expanded_tokens = pos_to_expert.shape[0]
        expanded_x = torch.empty(
            (num_expanded_tokens, self.hidden), dtype=x.dtype, device=x.device
        )
        # A zero-token batch leaves nothing to scatter, and the grid would be
        # sized only by num_expanded_tokens whose rows are all unassigned.
        if x.shape[0] > 0:
            self._kernel_fn()(
                x, None, expanded_x, None, token_topk_to_pos, pos_to_expert
            )
        return expanded_x


class MoeExpandToFusedWithSFKernel(Kernel):
    """Scatter quantized activations and their scale factors into fused layout.

    Moves the per-block scale factors alongside the activations in the same
    pass, so the expanded activation and its scale factor stay paired. The
    activation payload is copied without interpretation, so any 1-byte
    quantized encoding works -- ``float8_e4m3fn``, or fp4 packed two values per
    ``uint8``/``int8`` byte with ``hidden`` given as the packed byte count.

    Args:
        hidden: Hidden dimension H, in elements of ``dtype``.
        num_topk: Number of expert slots per token K.
        num_per_channels: Channels per scale-factor block; 32 or 128.
        dtype: Data type of the quantized activations.
        sf_dtype: Scale-factor dtype. ``torch.float32`` for plain scale
            factors, ``torch.int32`` for Packed UE8M0.
        use_tma_aligned_col_major_sf: Emit the SF output transposed as
            ``[hidden_sf, num_expanded_tokens]`` over a token dimension padded
            to a multiple of 4. Required for Packed UE8M0.
        config: Optional config dict.

    Example:
        >>> kernel = MoeExpandToFusedWithSFKernel(hidden=256, num_topk=2,
        ...                                       num_per_channels=128)
        >>> expanded_x, expanded_x_sf = kernel(x, x_sf, token_topk_to_pos,
        ...                                    pos_to_expert)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        num_per_channels: int,
        dtype: torch.dtype = torch.float8_e4m3fn,
        sf_dtype: torch.dtype = torch.float32,
        use_tma_aligned_col_major_sf: bool = False,
        num_threads: int = _DEFAULT_NUM_THREADS,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert num_per_channels in _SUPPORTED_NUM_PER_CHANNELS
        assert sf_dtype in (torch.float32, torch.int32)

        self.hidden = hidden
        self.num_topk = num_topk
        self.num_per_channels = num_per_channels
        self.dtype = dtype
        self.sf_dtype = sf_dtype
        self.num_threads = num_threads

        # An int32 SF buffer is Packed UE8M0 by construction: it is the only
        # encoding that puts four exponent bytes in one word, and the packed
        # form is defined solely for the TMA-aligned column-major layout.
        self.use_packed_ue8m0 = sf_dtype == torch.int32
        if self.use_packed_ue8m0:
            assert use_tma_aligned_col_major_sf, (
                "Packed UE8M0 scale factors require "
                "use_tma_aligned_col_major_sf=True"
            )
        self.use_tma_aligned_col_major_sf = use_tma_aligned_col_major_sf

        self.hidden_sf = _hidden_sf(hidden, num_per_channels, self.use_packed_ue8m0)

        self._kernel_fn = _expand_to_fused_kernel(
            hidden,
            num_topk,
            num_per_channels,
            use_tma_aligned_col_major_sf,
            self.use_packed_ue8m0,
            self.dtype_str,
            self.dtype_to_str(sf_dtype),
            num_threads,
        )

        self.init_config(config, tune=False)

    def forward(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run expand-to-fused on quantized activations.

        Args:
            x: [num_tokens, hidden] quantized token activations.
            x_sf: [num_tokens, hidden_sf] scale factors for ``x``.
            token_topk_to_pos: [num_tokens, num_topk] int32 map from a
                (token, slot) pair to its expanded position; negative marks a
                dropped slot.
            pos_to_expert: [num_expanded_tokens] int32 map from an expanded
                position to its expert; negative marks an unassigned position.

        Returns:
            A tuple ``(expanded_x, expanded_x_sf)``. ``expanded_x`` is
            [num_expanded_tokens, hidden]; ``expanded_x_sf`` is
            [num_expanded_tokens, hidden_sf], returned as a transposed view of
            the column-major buffer when
            ``use_tma_aligned_col_major_sf``. Unassigned positions are
            zero-filled in both.
        """
        assert x.is_cuda and x_sf.is_cuda
        assert token_topk_to_pos.is_cuda and pos_to_expert.is_cuda
        assert x.is_contiguous() and x_sf.is_contiguous()
        assert token_topk_to_pos.is_contiguous() and pos_to_expert.is_contiguous()
        assert token_topk_to_pos.dtype == torch.int32
        assert pos_to_expert.dtype == torch.int32
        assert x.dtype == self.dtype and x_sf.dtype == self.sf_dtype
        assert x.shape[1] == self.hidden
        assert x_sf.shape == (x.shape[0], self.hidden_sf)
        assert token_topk_to_pos.shape[1] == self.num_topk
        assert token_topk_to_pos.shape[0] == x.shape[0]

        num_expanded_tokens = pos_to_expert.shape[0]
        expanded_x = torch.empty(
            (num_expanded_tokens, self.hidden), dtype=x.dtype, device=x.device
        )

        if self.use_tma_aligned_col_major_sf:
            # Allocate over the padded token dimension the TMA layout needs,
            # then hand the kernel the exact column slice. The padding columns
            # stay untouched and are never returned.
            num_expanded_sf_tokens = _align(num_expanded_tokens, _SF_TOKEN_ALIGN)
            expanded_x_sf = torch.empty(
                (self.hidden_sf, num_expanded_sf_tokens),
                dtype=x_sf.dtype,
                device=x_sf.device,
            )[:, :num_expanded_tokens]
        else:
            expanded_x_sf = torch.empty(
                (num_expanded_tokens, self.hidden_sf),
                dtype=x_sf.dtype,
                device=x_sf.device,
            )

        if x.shape[0] > 0:
            self._kernel_fn()(
                x, x_sf, expanded_x, expanded_x_sf, token_topk_to_pos, pos_to_expert
            )

        if self.use_tma_aligned_col_major_sf:
            expanded_x_sf = expanded_x_sf.T
        return expanded_x, expanded_x_sf
