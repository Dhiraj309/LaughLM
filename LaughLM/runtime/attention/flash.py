"""
LaughLM/runtime/attention/flash.py

Production attention backend dispatch.

TPU:
    jax.nn.dot_product_attention
    -> Splash / fused XLA kernels

GPU:
    cuDNN FlashAttention

CPU:
    XLA fallback

This is the PRIMARY production backend.

online_softmax.py exists only for:
    - research
    - debugging
    - correctness validation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import (
    AttentionMaskSpec,
)

Array = jnp.ndarray


def flash_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
    *,
    block_q: int = 128,
    block_kv: int = 128,
) -> Array:
    """
    Production fused attention path.

    Uses backend-native fused kernels.
    """

    #
    # IMPORTANT:
    #
    # JAX handles:
    # - MHA
    # - GQA
    # - MQA
    #
    # natively when:
    #
    #   num_query_heads % num_kv_heads == 0
    #

    return jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,

        #
        # Runtime-generated causal masking
        #
        is_causal=True,
    )
