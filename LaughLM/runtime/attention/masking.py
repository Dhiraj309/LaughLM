"""
LaughLM/runtime/attention/masking.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import (
    AttentionMaskSpec,
    AttentionMaskType,
)

Array = jnp.ndarray


def causal_mask(
    q_len: int,
    kv_len: int,
    offset: int = 0,
) -> Array:
    """
    Standard causal mask.

    Shape:
        [q_len, kv_len]
    """

    q_idx = jax.lax.broadcasted_iota(
        jnp.int32,
        (q_len, kv_len),
        0,
    )

    kv_idx = jax.lax.broadcasted_iota(
        jnp.int32,
        (q_len, kv_len),
        1,
    )

    return kv_idx <= (q_idx + offset)


def sliding_window_mask(
    q_len: int,
    kv_len: int,
    window_size: int,
    offset: int = 0,
) -> Array:
    """
    Local sliding window causal mask.
    """

    q_idx = (
        jax.lax.broadcasted_iota(
            jnp.int32,
            (q_len, kv_len),
            0,
        )
        + offset
    )

    kv_idx = jax.lax.broadcasted_iota(
        jnp.int32,
        (q_len, kv_len),
        1,
    )

    lower = kv_idx > (q_idx - window_size)
    upper = kv_idx <= q_idx

    return lower & upper


def chunk_mask(
    q_len: int,
    kv_len: int,
    chunk_size: int,
    offset: int = 0,
) -> Array:
    """
    Chunked causal attention.

    Tokens only attend within chunk.
    """

    q_idx = (
        jax.lax.broadcasted_iota(
            jnp.int32,
            (q_len, kv_len),
            0,
        )
        + offset
    )

    kv_idx = jax.lax.broadcasted_iota(
        jnp.int32,
        (q_len, kv_len),
        1,
    )

    same_chunk = (
        q_idx // chunk_size
        ==
        kv_idx // chunk_size
    )

    causal = kv_idx <= q_idx

    return same_chunk & causal


def build_mask(
    q_len: int,
    kv_len: int,
    spec: AttentionMaskSpec,
    offset: int = 0,
) -> Array:

    if spec.mask_type == AttentionMaskType.FULL:
        return jnp.ones(
            (q_len, kv_len),
            dtype=jnp.bool_,
        )

    if spec.mask_type == AttentionMaskType.CAUSAL:
        return causal_mask(
            q_len,
            kv_len,
            offset,
        )

    if spec.mask_type == AttentionMaskType.SLIDING:
        return sliding_window_mask(
            q_len,
            kv_len,
            spec.sliding_window,
            offset,
        )

    if spec.mask_type == AttentionMaskType.CHUNK:
        return chunk_mask(
            q_len,
            kv_len,
            spec.chunk_size,
            offset,
        )

    raise ValueError(spec.mask_type)
