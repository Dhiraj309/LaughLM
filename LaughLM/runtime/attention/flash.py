"""
LaughLM/runtime/attention/flash.py
"""

from __future__ import annotations

import jax

from .types import (
    AttentionMaskSpec,
)

Array = object


def flash_attention(
    query,
    key,
    value,
    mask_spec: AttentionMaskSpec,
    *,
    block_q: int = 128,
    block_kv: int = 128,
):
    """
    Flash attention runtime wrapper.

    Current implementation:
        JAX fused SDPA.

    Future:
        - Splash TPU kernels
        - cuDNN FlashAttention
        - Triton
        - Pallas kernels
    """

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
    )
