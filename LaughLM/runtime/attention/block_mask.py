"""
LaughLM/runtime/attention/block_mask.py

Block-local mask generation for streaming attention.

Avoids materializing full [T, S] masks.
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import (
    AttentionMaskSpec,
    AttentionMaskType,
)

Array = jnp.ndarray


def make_block_mask(
    q_start: int,
    q_len: int,
    kv_start: int,
    kv_len: int,
    spec: AttentionMaskSpec,
) -> Array:
    """
    Create block-local boolean mask.

    Returns
    -------
    [q_len, kv_len] bool
    """

    q_idx = (
        jnp.arange(q_len, dtype=jnp.int32)
        + q_start
    )

    kv_idx = (
        jnp.arange(kv_len, dtype=jnp.int32)
        + kv_start
    )

    q_idx = q_idx[:, None]
    kv_idx = kv_idx[None, :]

    # ======================================================
    # Causal
    # ======================================================

    if spec.mask_type == AttentionMaskType.CAUSAL:

        return kv_idx <= q_idx

    # ======================================================
    # Sliding window
    # ======================================================

    if (
        spec.mask_type
        == AttentionMaskType.SLIDING_WINDOW
    ):

        if spec.sliding_window is None:
            raise ValueError(
                "sliding_window required"
            )

        return (
            (kv_idx <= q_idx)
            &
            (
                kv_idx
                >= (
                    q_idx
                    - spec.sliding_window
                )
            )
        )

    # ======================================================
    # Chunked
    # ======================================================

    if (
        spec.mask_type
        == AttentionMaskType.CHUNKED
    ):

        if spec.chunk_size is None:
            raise ValueError(
                "chunk_size required"
            )

        q_chunk = (
            q_idx // spec.chunk_size
        )

        kv_chunk = (
            kv_idx // spec.chunk_size
        )

        return q_chunk == kv_chunk

    # ======================================================
    # Full
    # ======================================================

    if spec.mask_type == AttentionMaskType.FULL:

        return jnp.ones(
            (q_len, kv_len),
            dtype=jnp.bool_,
        )

    raise ValueError(
        f"Unknown mask type: {spec.mask_type}"
    )
