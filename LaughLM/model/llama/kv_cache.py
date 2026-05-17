"""
LaughLM/model/llama/kv_cache.py

Canonical static KV cache for Llama inference.

Design goals
------------
- deterministic cache semantics
- HF-compatible static cache behavior
- backend-agnostic implementation
- compile-friendly static shapes
- exact decode parity

Cache layout
-------------
key:
    [B, max_seq_len, KVH, Dh]

value:
    [B, max_seq_len, KVH, Dh]

IMPORTANT
---------
The cache always stores FULL static tensors.

Attention masking and slicing determine visible tokens.

The cache itself is never dynamically resized.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class KVCache(NamedTuple):

    key: jnp.ndarray
    value: jnp.ndarray

    cache_position: jnp.ndarray


def init_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=jnp.bfloat16,
) -> KVCache:
    """
    Initialize static KV cache.

    Shapes
    ------
    key:
        [B, S, KVH, Dh]

    value:
        [B, S, KVH, Dh]
    """

    shape = (
        batch_size,
        max_seq_len,
        num_kv_heads,
        head_dim,
    )

    return KVCache(
        key=jnp.zeros(shape, dtype=dtype),
        value=jnp.zeros(shape, dtype=dtype),
        cache_position=jnp.array(0, dtype=jnp.int32),
    )


def update_kv_cache(
    cache: KVCache,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
) -> tuple[
    KVCache,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Update static KV cache.

    Parameters
    ----------
    key_states:
        [B, T, KVH, Dh]

    value_states:
        [B, T, KVH, Dh]

    Returns
    -------
    updated_cache:
        Updated static cache

    full_key_states:
        [B, S, KVH, Dh]

    full_value_states:
        [B, S, KVH, Dh]

    IMPORTANT
    ---------
    Returns FULL cache tensors.

    Attention logic is responsible for:
    - causal masking
    - valid-length restriction
    - decode visibility
    """

    start = cache.cache_position

    seq_len = key_states.shape[1]

    updated_keys = jax.lax.dynamic_update_slice(
        cache.key,
        key_states.astype(cache.key.dtype),
        (
            0,
            start,
            0,
            0,
        ),
    )

    updated_values = jax.lax.dynamic_update_slice(
        cache.value,
        value_states.astype(cache.value.dtype),
        (
            0,
            start,
            0,
            0,
        ),
    )

    updated_cache = KVCache(
        key=updated_keys,
        value=updated_values,
        cache_position=start + seq_len,
    )

    return (
        updated_cache,
        updated_keys,
        updated_values,
    )


def get_cache_length(
    cache: KVCache,
) -> jnp.ndarray:
    """
    Current valid cache length.

    Returns
    -------
    scalar int32
    """

    return cache.cache_position