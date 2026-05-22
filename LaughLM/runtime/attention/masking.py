"""
LaughLM/runtime/attention/masking.py
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import (
    AttentionMaskSpec,
    AttentionMaskType,
)

Array = jnp.ndarray


def build_causal_mask(
    q_len: int,
    kv_len: int,
) -> Array:

    q_idx = jnp.arange(q_len)[:, None]
    kv_idx = jnp.arange(kv_len)[None, :]

    offset = kv_len - q_len

    return q_idx + offset >= kv_idx


def build_full_mask(
    q_len: int,
    kv_len: int,
) -> Array:

    return jnp.ones(
        (q_len, kv_len),
        dtype=jnp.bool_,
    )


def build_sliding_window_mask(
    q_len: int,
    kv_len: int,
    window: int,
) -> Array:

    q_idx = jnp.arange(q_len)[:, None]
    kv_idx = jnp.arange(kv_len)[None, :]

    offset = kv_len - q_len

    causal = (
        q_idx + offset >= kv_idx
    )

    local = (
        q_idx + offset - kv_idx
    ) < window

    return causal & local


def build_chunk_mask(
    q_len: int,
    kv_len: int,
    chunk_size: int,
) -> Array:

    q_idx = jnp.arange(q_len)[:, None]
    kv_idx = jnp.arange(kv_len)[None, :]

    offset = kv_len - q_len

    q_abs = q_idx + offset

    causal = q_abs >= kv_idx

    same_chunk = (
        q_abs // chunk_size
        ==
        kv_idx // chunk_size
    )

    return causal & same_chunk


def build_mask(
    q_len: int,
    kv_len: int,
    spec: AttentionMaskSpec,
) -> Array:

    if spec.mask_type == AttentionMaskType.CAUSAL:

        return build_causal_mask(
            q_len,
            kv_len,
        )

    if spec.mask_type == AttentionMaskType.FULL:

        return build_full_mask(
            q_len,
            kv_len,
        )

    if (
        spec.mask_type
        ==
        AttentionMaskType.SLIDING_WINDOW
    ):

        if spec.sliding_window is None:
            raise ValueError(
                "sliding_window missing"
            )

        return build_sliding_window_mask(
            q_len,
            kv_len,
            spec.sliding_window,
        )

    if spec.mask_type == AttentionMaskType.CHUNK:

        if spec.chunk_size is None:
            raise ValueError(
                "chunk_size missing"
            )

        return build_chunk_mask(
            q_len,
            kv_len,
            spec.chunk_size,
        )

    raise ValueError(
        f"Unknown mask type: {spec.mask_type}"
    )
