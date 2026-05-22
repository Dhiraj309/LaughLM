"""
LaughLM/runtime/attention/reference.py
"""

from __future__ import annotations

import jax.numpy as jnp

from .masking import build_mask
from .types import (
    AttentionMaskSpec,
    DEFAULT_MASK_VALUE,
)

Array = jnp.ndarray


def reference_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
) -> Array:
    """
    Reference GQA/MHA attention.

    query:
        [B, T, Hq, D]

    key/value:
        [B, S, Hkv, D]
    """

    B, T, Hq, D = query.shape
    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    #
    # Reshape query into GQA groups
    #

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

    #
    # QK
    #

    logits = jnp.einsum(
        "bthgd,bshd->bhgts",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )

    #
    # Mask
    #

    mask = build_mask(
        q_len=T,
        kv_len=S,
        spec=mask_spec,
    )

    logits = logits + jnp.where(
        mask,
        0.0,
        DEFAULT_MASK_VALUE,
    )[None, None, None, :, :]

    #
    # Softmax
    #

    probs = jax.nn.softmax(
        logits,
        axis=-1,
    )

    #
    # OV
    #

    out = jnp.einsum(
        "bhgts,bshd->bthgd",
        probs.astype(value.dtype),
        value,
    )

    out = out.reshape(
        B,
        T,
        Hq,
        D,
    )

    return out
