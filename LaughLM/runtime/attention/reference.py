"""
LaughLM/runtime/attention/reference.py
"""

from __future__ import annotations

import jax
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

    # ======================================================
    # Reshape query into GQA groups
    #
    # [B, T, Hq, D]
    # ->
    # [B, T, Hkv, G, D]
    # ======================================================

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

    # ======================================================
    # QK
    #
    # output:
    # [B, Hkv, G, T, S]
    # ======================================================

    logits = jnp.einsum(
        "bthgd,bshd->bhgts",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )

    logits = logits * (D ** -0.5)

    # ======================================================
    # Mask
    # ======================================================

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

    # ======================================================
    # Softmax
    # ======================================================

    probs = jax.nn.softmax(
        logits,
        axis=-1,
    )

    probs = probs.astype(
        value.dtype
    )

    # ======================================================
    # OV
    #
    # [B, Hkv, G, T, D]
    # ======================================================

    out = jnp.einsum(
        "bhgts,bshd->bthgd",
        probs,
        value,
        preferred_element_type=jnp.float32,
    )

    # ======================================================
    # Restore head layout
    # ======================================================

    out = out.reshape(
        B,
        T,
        Hq,
        D,
    )

    return out
