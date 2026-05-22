"""
LaughLM/runtime/attention/decode.py
"""

from __future__ import annotations

import jax

from .types import (
    AttentionMaskSpec,
)

Array = object


def decode_attention(
    query,
    key,
    value,
    mask_spec: AttentionMaskSpec,
    *,
    block_kv: int = 128,
):
    """
    Decode-specialized attention.

    Optimized for:
        query length == 1

    KV cache already guarantees causal validity,
    so explicit causal masking is unnecessary.

    Uses fused JAX SDPA kernel instead of
    manual online softmax implementation.
    """

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
    )
