"""MoE expand-to-fused ops: scatter tokens into the fused expert layout.

Provides:
  - MoeExpandToFusedFwdOp: expanded_x[p] = x[t] for every routed (t, k) -> p
  - MoeExpandToFusedWithSFFwdOp: the same scatter for quantized activations,
    carrying per-block scale factors alongside the data
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.expand_to_fused import (
    MoeExpandToFusedKernel,
    MoeExpandToFusedWithSFKernel,
    select_expand_to_fused_num_threads,
)

from ..op_base import Op

__all__ = ["MoeExpandToFusedFwdOp", "MoeExpandToFusedWithSFFwdOp"]

# Scale-factor block widths declared by the manifest shape rules.
_SUPPORTED_NUM_PER_CHANNELS = (32, 128)

# UE8M0 exponent bytes packed per int32 word.
_UE8M0_PER_INT32 = 4


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def _check_routing_inputs(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    tensors: Tuple[Tuple[str, torch.Tensor], ...],
) -> Tuple[int, int, int, int]:
    """Validate the routing-plan contract shared by both expand-to-fused ops.

    Args:
        x: Activation tensor, the device and token-count reference.
        token_topk_to_pos: [T, K] int32 routing map.
        pos_to_expert: [X] int32 position-to-expert map.
        tensors: All (name, tensor) pairs to residency-check, in signature
            order.

    Returns:
        ``(num_tokens, hidden, num_topk, num_expanded_tokens)``.

    Raises:
        ValueError: If any residency, rank, or shape rule is violated.
    """
    for name, t in tensors:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != x.device:
            raise ValueError(
                f"Expected all inputs on the same device, got {x.device} "
                f"for x and {t.device} for {name}"
            )
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D [T, H], got {x.ndim}D")
    if token_topk_to_pos.ndim != 2:
        raise ValueError(
            f"Expected token_topk_to_pos to be 2D [T, K], "
            f"got {token_topk_to_pos.ndim}D"
        )
    if pos_to_expert.ndim != 1:
        raise ValueError(
            f"Expected pos_to_expert to be 1D [X], got {pos_to_expert.ndim}D"
        )

    num_tokens, hidden = x.shape
    if num_tokens <= 0:
        raise ValueError(f"Expected x.shape[0] > 0, got {num_tokens}")
    if token_topk_to_pos.shape[0] != num_tokens:
        raise ValueError(
            f"Expected token_topk_to_pos.shape[0] == x.shape[0] "
            f"({num_tokens}), got {token_topk_to_pos.shape[0]}"
        )
    num_topk = token_topk_to_pos.shape[1]
    if num_topk <= 0:
        raise ValueError(
            f"Expected token_topk_to_pos.shape[1] > 0, got {num_topk}"
        )
    return num_tokens, hidden, num_topk, pos_to_expert.shape[0]


def _check_index_dtypes(
    token_topk_to_pos: torch.Tensor, pos_to_expert: torch.Tensor
) -> None:
    """Raise if either routing map is not int32."""
    if token_topk_to_pos.dtype != torch.int32:
        raise ValueError(
            f"Expected token_topk_to_pos.dtype torch.int32, "
            f"got {token_topk_to_pos.dtype}"
        )
    if pos_to_expert.dtype != torch.int32:
        raise ValueError(
            f"Expected pos_to_expert.dtype torch.int32, got {pos_to_expert.dtype}"
        )


class MoeExpandToFusedFwdOp(Op):
    """Expand token activations into the fused expert layout.

    ``token_topk_to_pos`` maps each (token, expert-slot) pair to a row of the
    expanded buffer; ``pos_to_expert`` labels each expanded row with its
    expert. Both use a negative entry as the sentinel for a dropped slot /
    unassigned position, so the expanded row count is data-dependent and is
    taken from ``pos_to_expert`` rather than committed at construction.

    Args:
        dtype: Optional committed activation dtype (bf16 or fp16). Preferred
            API infers it from ``x``.
        kernel_map: Optional override for kernel dispatch.

    Example:
        >>> op = MoeExpandToFusedFwdOp()
        >>> expanded_x = op(x, token_topk_to_pos, pos_to_expert)
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    # The manifest entry declares no static_dims: hidden and num_topk are read
    # from the inputs at forward time, and the token counts are TileLang
    # dynamic symbols. See _cache_key for the resulting kernel-cache
    # projection.
    _static_axes: frozenset[tuple[int, int]] = frozenset()

    def __init__(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.dtype = dtype
        self._committed_dtype = dtype

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[Tuple[int, int, torch.dtype, int], Kernel] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"expand_to_fused_kernel": MoeExpandToFusedKernel}

    def _cache_key(
        self,
        x_shape: tuple,
        token_topk_to_pos_shape: tuple,
        pos_to_expert_shape: tuple,
    ) -> Tuple[int, int]:
        """Project input shapes onto what the compiled kernel depends on.

        Only ``hidden`` and ``num_topk`` are baked into the prim_func; the two
        token counts are ``T.dynamic`` symbols, so batch size does not
        fragment the cache.
        """
        return (x_shape[1], token_topk_to_pos_shape[1])

    def _infer_output_shapes(
        self,
        x_shape: tuple,
        token_topk_to_pos_shape: tuple,
        pos_to_expert_shape: tuple,
    ) -> Dict[str, tuple]:
        return {"expanded_x": (pos_to_expert_shape[0], x_shape[1])}

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> None:
        if x.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"Expected x.dtype in [{names}], got {x.dtype}")
        _check_index_dtypes(token_topk_to_pos, pos_to_expert)

    def eval_roofline(self) -> tuple[int, int]:
        if (
            not hasattr(self, "x_shape")
            or not hasattr(self, "token_topk_to_pos_shape")
            or not hasattr(self, "pos_to_expert_shape")
            or self.dtype is None
        ):
            raise ValueError(
                "MoeExpandToFusedFwdOp.eval_roofline() requires a prior forward() "
                "to bind x_shape, token_topk_to_pos_shape, pos_to_expert_shape, "
                "and dtype"
            )
        num_tokens, hidden = self.x_shape
        num_topk = self.token_topk_to_pos_shape[1]
        num_expanded_tokens = self.pos_to_expert_shape[0]
        elem_bytes = self.dtype.itemsize
        nbytes = (
            num_tokens * hidden * elem_bytes
            + num_tokens * num_topk * 4
            + num_expanded_tokens * 4
            + num_expanded_tokens * hidden * elem_bytes
        )
        return 0, int(nbytes)

    def _get_kernel(
        self,
        hidden: int,
        num_topk: int,
        dtype: torch.dtype,
        num_threads: int,
    ) -> Kernel:
        key = (hidden, num_topk, dtype, num_threads)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["expand_to_fused_kernel"](
                hidden, num_topk, dtype, num_threads
            )
        return self._kernel_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run expand-to-fused.

        Args:
            x: [num_tokens, hidden] token activations (bf16/fp16).
            token_topk_to_pos: [num_tokens, num_topk] int32 map from a
                (token, slot) pair to its expanded position; negative marks a
                dropped slot.
            pos_to_expert: [num_expanded_tokens] int32 map from an expanded
                position to its expert; negative marks an unassigned position.

        Returns:
            expanded_x: [num_expanded_tokens, hidden] activations in fused
                expert layout. Unassigned positions are zero-filled.
        """
        self._validate_dtypes(x, token_topk_to_pos, pos_to_expert)
        _, hidden, num_topk, _ = _check_routing_inputs(
            x,
            token_topk_to_pos,
            pos_to_expert,
            (
                ("x", x),
                ("token_topk_to_pos", token_topk_to_pos),
                ("pos_to_expert", pos_to_expert),
            ),
        )
        if self._committed_dtype is not None and x.dtype != self._committed_dtype:
            raise ValueError(
                f"Expected x.dtype {self._committed_dtype}, got {x.dtype}"
            )

        x = x.contiguous()
        token_topk_to_pos = token_topk_to_pos.contiguous()
        pos_to_expert = pos_to_expert.contiguous()

        self.dtype = x.dtype
        self.x_shape = tuple(x.shape)
        self.token_topk_to_pos_shape = tuple(token_topk_to_pos.shape)
        self.pos_to_expert_shape = tuple(pos_to_expert.shape)

        num_threads = select_expand_to_fused_num_threads(
            hidden, x.shape[0], has_scale_factors=False
        )
        kernel = self._get_kernel(hidden, num_topk, x.dtype, num_threads)
        return kernel(x, token_topk_to_pos, pos_to_expert)


class MoeExpandToFusedWithSFFwdOp(Op):
    """Expand quantized activations and their scale factors into fused layout.

    The routing contract is identical to :class:`MoeExpandToFusedFwdOp`; this
    variant additionally moves the per-block scale factors (SF) so each
    expanded row keeps the scale that decodes it.

    ``num_per_channels`` sets the SF block width, giving
    ``S = ceil(H / num_per_channels)`` float32 scale factors per row. Passing
    an int32 ``x_sf`` selects Packed UE8M0, where four UE8M0 exponent bytes
    share one word and ``S`` shrinks by a further factor of 4; that encoding is
    only defined for the TMA-aligned column-major SF layout.

    Args:
        num_per_channels: Channels per scale-factor block; 32 or 128.
        use_tma_aligned_col_major_sf: Return ``expanded_x_sf`` as a transposed
            view of a column-major buffer whose token dimension is padded to a
            multiple of 4, which is what the downstream TMA-based grouped GEMM
            reads. Required for Packed UE8M0.
        dtype: Optional committed activation dtype. Preferred API infers it
            from ``x``.
        sf_dtype: Optional committed scale-factor dtype. Preferred API infers
            it from ``x_sf``.
        kernel_map: Optional override for kernel dispatch.

    Example:
        >>> op = MoeExpandToFusedWithSFFwdOp(num_per_channels=128)
        >>> expanded_x, expanded_x_sf = op(x, x_sf, token_topk_to_pos,
        ...                                pos_to_expert)
    """

    # The kernel copies the activation payload without interpreting it, so fp8
    # and packed-fp4 byte storage both run through the same path unchanged.
    _SUPPORTED_DTYPES = (torch.float8_e4m3fn, torch.uint8, torch.int8)
    _SUPPORTED_SF_DTYPES = (torch.float32, torch.int32)

    # As in the unquantized op: hidden, num_topk and the SF width are read from
    # the inputs at forward time, and both token counts are dynamic symbols.
    _static_axes: frozenset[tuple[int, int]] = frozenset()

    def __init__(
        self,
        *,
        num_per_channels: int,
        use_tma_aligned_col_major_sf: bool = False,
        dtype: Optional[torch.dtype] = None,
        sf_dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        if num_per_channels not in _SUPPORTED_NUM_PER_CHANNELS:
            raise ValueError(
                f"Expected num_per_channels in "
                f"{list(_SUPPORTED_NUM_PER_CHANNELS)}, got {num_per_channels}"
            )
        self.num_per_channels = num_per_channels
        self.use_tma_aligned_col_major_sf = use_tma_aligned_col_major_sf

        self.dtype = dtype
        self.sf_dtype = sf_dtype
        self._committed_dtype = dtype
        self._committed_sf_dtype = sf_dtype

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[
            Tuple[int, int, torch.dtype, torch.dtype, int], Kernel
        ] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"expand_to_fused_with_sf_kernel": MoeExpandToFusedWithSFKernel}

    def _expected_hidden_sf(self, hidden: int, sf_dtype: torch.dtype) -> int:
        """Return the SF column count the manifest shape rules require."""
        hidden_sf = _ceil_div(hidden, self.num_per_channels)
        if sf_dtype == torch.int32:
            hidden_sf = _ceil_div(hidden_sf, _UE8M0_PER_INT32)
        return hidden_sf

    def _cache_key(
        self,
        x_shape: tuple,
        x_sf_shape: tuple,
        token_topk_to_pos_shape: tuple,
        pos_to_expert_shape: tuple,
    ) -> Tuple[int, int]:
        """Project input shapes onto what the compiled kernel depends on.

        ``hidden`` and ``num_topk`` are baked into the prim_func; the SF width
        is derived from ``hidden`` and the ctor-committed ``num_per_channels``,
        so it adds no independent axis. Both token counts are ``T.dynamic``
        symbols.
        """
        return (x_shape[1], token_topk_to_pos_shape[1])

    def _infer_output_shapes(
        self,
        x_shape: tuple,
        x_sf_shape: tuple,
        token_topk_to_pos_shape: tuple,
        pos_to_expert_shape: tuple,
    ) -> Dict[str, tuple]:
        num_expanded_tokens = pos_to_expert_shape[0]
        return {
            "expanded_x": (num_expanded_tokens, x_shape[1]),
            "expanded_x_sf": (num_expanded_tokens, x_sf_shape[1]),
        }

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> None:
        if x.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"Expected x.dtype in [{names}], got {x.dtype}")
        if x_sf.dtype not in self._SUPPORTED_SF_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_SF_DTYPES)
            raise ValueError(f"Expected x_sf.dtype in [{names}], got {x_sf.dtype}")
        _check_index_dtypes(token_topk_to_pos, pos_to_expert)

    def eval_roofline(self) -> tuple[int, int]:
        if (
            not hasattr(self, "x_shape")
            or not hasattr(self, "x_sf_shape")
            or not hasattr(self, "token_topk_to_pos_shape")
            or not hasattr(self, "pos_to_expert_shape")
            or self.dtype is None
        ):
            raise ValueError(
                "MoeExpandToFusedWithSFFwdOp.eval_roofline() requires a prior "
                "forward() to bind x_shape, x_sf_shape, token_topk_to_pos_shape, "
                "pos_to_expert_shape, and dtype"
            )
        num_tokens, hidden = self.x_shape
        hidden_sf = self.x_sf_shape[1]
        num_topk = self.token_topk_to_pos_shape[1]
        num_expanded_tokens = self.pos_to_expert_shape[0]
        elem_bytes = self.dtype.itemsize
        # Scale factors are 4-byte float32 or packed int32 words either way.
        nbytes = (
            num_tokens * hidden * elem_bytes
            + num_tokens * hidden_sf * 4
            + num_tokens * num_topk * 4
            + num_expanded_tokens * 4
            + num_expanded_tokens * hidden * elem_bytes
            + num_expanded_tokens * hidden_sf * 4
        )
        return 0, int(nbytes)

    def _get_kernel(
        self,
        hidden: int,
        num_topk: int,
        dtype: torch.dtype,
        sf_dtype: torch.dtype,
        num_threads: int,
    ) -> Kernel:
        key = (hidden, num_topk, dtype, sf_dtype, num_threads)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[
                "expand_to_fused_with_sf_kernel"
            ](
                hidden,
                num_topk,
                self.num_per_channels,
                dtype,
                sf_dtype,
                self.use_tma_aligned_col_major_sf,
                num_threads,
            )
        return self._kernel_cache[key]

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
            x_sf: [num_tokens, hidden_sf] scale factors for ``x``; float32, or
                int32 for Packed UE8M0.
            token_topk_to_pos: [num_tokens, num_topk] int32 map from a
                (token, slot) pair to its expanded position; negative marks a
                dropped slot.
            pos_to_expert: [num_expanded_tokens] int32 map from an expanded
                position to its expert; negative marks an unassigned position.

        Returns:
            A tuple ``(expanded_x, expanded_x_sf)`` of shapes
            [num_expanded_tokens, hidden] and [num_expanded_tokens, hidden_sf].
            Unassigned positions are zero-filled in both. Under
            ``use_tma_aligned_col_major_sf`` the SF result is a transposed view
            of a column-major buffer, so it is not row-major contiguous.
        """
        self._validate_dtypes(x, x_sf, token_topk_to_pos, pos_to_expert)
        _, hidden, num_topk, _ = _check_routing_inputs(
            x,
            token_topk_to_pos,
            pos_to_expert,
            (
                ("x", x),
                ("x_sf", x_sf),
                ("token_topk_to_pos", token_topk_to_pos),
                ("pos_to_expert", pos_to_expert),
            ),
        )
        if x_sf.ndim != 2:
            raise ValueError(f"Expected x_sf to be 2D [T, S], got {x_sf.ndim}D")
        if x_sf.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected x_sf.shape[0] == x.shape[0] ({x.shape[0]}), "
                f"got {x_sf.shape[0]}"
            )
        # Packed UE8M0 only exists in the TMA-aligned column-major layout: the
        # packing groups four scale factors along the token-major axis.
        if x_sf.dtype == torch.int32 and not self.use_tma_aligned_col_major_sf:
            raise ValueError(
                "int32 x_sf selects Packed UE8M0, which requires "
                "use_tma_aligned_col_major_sf=True"
            )
        expected_hidden_sf = self._expected_hidden_sf(hidden, x_sf.dtype)
        if x_sf.shape[1] != expected_hidden_sf:
            raise ValueError(
                f"Expected x_sf.shape[1] == {expected_hidden_sf} for hidden "
                f"{hidden}, num_per_channels {self.num_per_channels} and "
                f"x_sf.dtype {x_sf.dtype}, got {x_sf.shape[1]}"
            )
        if self._committed_dtype is not None and x.dtype != self._committed_dtype:
            raise ValueError(
                f"Expected x.dtype {self._committed_dtype}, got {x.dtype}"
            )
        if (
            self._committed_sf_dtype is not None
            and x_sf.dtype != self._committed_sf_dtype
        ):
            raise ValueError(
                f"Expected x_sf.dtype {self._committed_sf_dtype}, got {x_sf.dtype}"
            )

        x = x.contiguous()
        x_sf = x_sf.contiguous()
        token_topk_to_pos = token_topk_to_pos.contiguous()
        pos_to_expert = pos_to_expert.contiguous()

        self.dtype = x.dtype
        self.sf_dtype = x_sf.dtype
        self.x_shape = tuple(x.shape)
        self.x_sf_shape = tuple(x_sf.shape)
        self.token_topk_to_pos_shape = tuple(token_topk_to_pos.shape)
        self.pos_to_expert_shape = tuple(pos_to_expert.shape)

        num_threads = select_expand_to_fused_num_threads(
            hidden, x.shape[0], has_scale_factors=True
        )
        kernel = self._get_kernel(hidden, num_topk, x.dtype, x_sf.dtype, num_threads)
        return kernel(x, x_sf, token_topk_to_pos, pos_to_expert)
