"""
LaughLM/runtime/attention/backend.py

Unified attention runtime dispatcher.

Backend semantics
─────────────────────────────────────

REFERENCE:
    naive correctness implementation

ONLINE:
    research/debug online softmax kernel

FLASH:
    production TPU SplashAttention backend

DECODE:
    specialized single-token decode path
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import (
    AttentionBackend,
    AttentionMaskSpec,
)

from .reference import (
    reference_attention,
)

from .decode import (
    decode_attention,
)

from .online_softmax import (
    online_attention,
)

from .splash import (
    splash_attention,
)

Array = jnp.ndarray


def apply_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
    backend: AttentionBackend,
    *,
    block_q: int = 128,
    block_kv: int = 128,
) -> Array:
    """
    Unified attention runtime dispatcher.

    Attention paths
    ─────────────────────────────────

    REFERENCE:
        correctness validation

    ONLINE:
        research/debug implementation

    FLASH:
        production TPU SplashAttention

    DECODE:
        specialized autoregressive decode
    """

    # ======================================================
    # Decode specialization
    # ======================================================

    #
    # IMPORTANT:
    #
    # Single-token decode should NEVER use
    # long-sequence training kernels.
    #
    # SplashAttention block tiling becomes
    # pathological for T=1.
    #

    if query.shape[1] == 1:

        if backend in (
            AttentionBackend.DECODE,
            AttentionBackend.FLASH,
            AttentionBackend.ONLINE,
        ):

            return decode_attention(
                query,
                key,
                value,
                mask_spec,
                block_kv=block_kv,
            )

    # ======================================================
    # Reference backend
    # ======================================================

    if backend == AttentionBackend.REFERENCE:

        return reference_attention(
            query,
            key,
            value,
            mask_spec,
        )

    # ======================================================
    # Research backend
    # ======================================================

    if backend == AttentionBackend.ONLINE:

        return online_attention(
            query,
            key,
            value,
            mask_spec,
            block_q=block_q,
            block_kv=block_kv,
        )

    # ======================================================
    # Production fused TPU backend
    # ======================================================

    if backend == AttentionBackend.FLASH:

        return splash_attention(
            query,
            key,
            value,
            mask_spec,
        )

    raise ValueError(
        f"Unknown backend: {backend}"
    )
