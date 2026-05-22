"""
LaughLM/runtime/attention/backend.py
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

from .flash import (
    flash_attention,
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
    """

    # ======================================================
    # Decode specialization
    # ======================================================

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
    # Reference
    # ======================================================

    if backend == AttentionBackend.REFERENCE:

        return reference_attention(
            query,
            key,
            value,
            mask_spec,
        )

    # ======================================================
    # ONLINE / FLASH
    # ======================================================

    if backend in (
        AttentionBackend.ONLINE,
        AttentionBackend.FLASH,
    ):

        return flash_attention(
            query,
            key,
            value,
            mask_spec,
            block_q=block_q,
            block_kv=block_kv,
        )

    raise ValueError(
        f"Unknown backend: {backend}"
    )
