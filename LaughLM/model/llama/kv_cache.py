"""
LaughLM/model/llama/kv_cache.py

Canonical static KV cache for Llama-style autoregressive decoding.

Design goals:
- deterministic decode semantics
- static shapes for JAX/XLA
- HF-compatible cache behavior
- backend-agnostic semantics
- future-compatible with:
    - GSPMD
    - paged attention
    - continuous batching
    - vLLM-style schedulers

Tensor conventions
------------------
key/value:
    [batch, max_seq_len, num_key_value_heads, head_dim]

query:
    [batch, query_seq_len, num_attention_heads, head_dim]

Valid cache region:
    [:, :cache_position]

Important invariants
--------------------
- RoPE is applied BEFORE cache insertion
- cache_position is the next write index
- update() returns ONLY valid cache slices
- attention never sees invalid cache regions
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCache:

    key: jnp.ndarray

    value: jnp.ndarray

    cache_position: jnp.ndarray


def create_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> KVCache:
    """
    Create empty static KV cache.

    Returns
    -------
    KVCache
    """

    shape = (
        batch_size,
        max_seq_len,
        num_key_value_heads,
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
    Append new KV states into static cache.

    Parameters
    ----------
    key_states:
        [B, T_new, KVH, Dh]

    value_states:
        [B, T_new, KVH, Dh]

    Returns
    -------
    updated_cache

    valid_key_states:
        [B, cache_seq_len, KVH, Dh]

    valid_value_states:
        [B, cache_seq_len, KVH, Dh]

    Important
    ---------
    Returned tensors contain ONLY valid cache entries.
    Invalid/uninitialized cache regions are never exposed
    to attention.
    """

    insert_pos = cache.cache_position

    seq_len = key_states.shape[1]

    new_key = jax.lax.dynamic_update_slice(
        cache.key,
        key_states.astype(cache.key.dtype),
        (0, insert_pos, 0, 0),
    )

    new_value = jax.lax.dynamic_update_slice(
        cache.value,
        value_states.astype(cache.value.dtype),
        (0, insert_pos, 0, 0),
    )

    new_position = insert_pos + seq_len

    updated_cache = KVCache(
        key=new_key,
        value=new_value,
        cache_position=new_position,
    )

    valid_key_states = new_key[
        :,
        :new_position,
        :,
        :,
    ]

    valid_value_states = new_value[
        :,
        :new_position,
        :,
        :,
    ]

    return (
        updated_cache,
        valid_key_states,
        valid_value_states,
    )