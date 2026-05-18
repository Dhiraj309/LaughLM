"""
LaughLM/model/llama/kv_cache.py

Canonical static KV cache for Llama inference.

Frontier-grade additions
────────────────────────────────────────────
1. Static-shape cache semantics
2. Deterministic append-only updates
3. GSPMD-compatible cache layouts
4. Compile-safe cache indexing
5. Explicit cache-length semantics
6. Future-ready decode specialization
7. Logical KV-cache constraints

Cache layout
────────────
key/value:
    [B, S, KVH, Dh]

IMPORTANT
─────────
The cache is ALWAYS statically allocated.

Only cache_position changes dynamically.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from LaughLM.distributed.sharding import (
    constrain_kv_cache,
)


# ─────────────────────────────────────────────
# KV cache container
# ─────────────────────────────────────────────

class KVCache(NamedTuple):

    key: jnp.ndarray

    value: jnp.ndarray

    #
    # Scalar int32
    #
    # Current valid sequence length.
    #
    cache_position: jnp.ndarray


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

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
    ─────────────────────────────────────────
    key/value:
        [B, S, KVH, Dh]
    """

    shape = (
        batch_size,
        max_seq_len,
        num_kv_heads,
        head_dim,
    )

    key = jnp.zeros(
        shape,
        dtype=dtype,
    )

    value = jnp.zeros(
        shape,
        dtype=dtype,
    )

    key = constrain_kv_cache(
        key
    )

    value = constrain_kv_cache(
        value
    )

    return KVCache(
        key=key,
        value=value,
        cache_position=jnp.asarray(
            0,
            dtype=jnp.int32,
        ),
    )


# ─────────────────────────────────────────────
# Cache update
# ─────────────────────────────────────────────

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
    Append KV states into static cache.

    Parameters
    ─────────────────────────────────────────
    key_states:
        [B, T, KVH, Dh]

    value_states:
        [B, T, KVH, Dh]

    Returns
    ─────────────────────────────────────────
    updated_cache

    full_key_states:
        [B, S, KVH, Dh]

    full_value_states:
        [B, S, KVH, Dh]
    """

    start = cache.cache_position

    #
    # Static compile-time sequence length.
    #
    seq_len = key_states.shape[1]

    max_seq_len = cache.key.shape[1]

    # --------------------------------------------------
    # Runtime bounds safety
    # --------------------------------------------------

    if (
        isinstance(seq_len, int)
        and seq_len > max_seq_len
    ):
        raise ValueError(
            "Incoming KV states exceed "
            "cache capacity."
        )

    # --------------------------------------------------
    # Overflow assertion
    # --------------------------------------------------

    overflow = (
        start + seq_len
        > max_seq_len
    )

    def overflow_error(_):

        raise ValueError(
            "KV cache overflow.\n"
            f"start={start}\n"
            f"seq_len={seq_len}\n"
            f"max_seq_len={max_seq_len}"
        )

    jax.debug.callback(
        lambda x: overflow_error(x)
        if bool(x)
        else None,
        overflow,
    )

    # --------------------------------------------------
    # Cast to cache dtype
    # --------------------------------------------------

    key_states = key_states.astype(
        cache.key.dtype
    )

    value_states = value_states.astype(
        cache.value.dtype
    )

    # --------------------------------------------------
    # Static append update
    # --------------------------------------------------

    updated_keys = (
        jax.lax.dynamic_update_slice(
            cache.key,
            key_states,
            (
                0,
                start,
                0,
                0,
            ),
        )
    )

    updated_values = (
        jax.lax.dynamic_update_slice(
            cache.value,
            value_states,
            (
                0,
                start,
                0,
                0,
            ),
        )
    )

    # --------------------------------------------------
    # Logical constraints
    # --------------------------------------------------

    updated_keys = constrain_kv_cache(
        updated_keys
    )

    updated_values = constrain_kv_cache(
        updated_values
    )

    # --------------------------------------------------
    # Updated cache
    # --------------------------------------------------

    updated_cache = KVCache(
        key=updated_keys,
        value=updated_values,
        cache_position=(
            start
            + jnp.asarray(
                seq_len,
                dtype=jnp.int32,
            )
        ),
    )

    return (
        updated_cache,
        updated_keys,
        updated_values,
    )


# ─────────────────────────────────────────────
# Cache length helper
# ─────────────────────────────────────────────

def get_cache_length(
    cache: KVCache,
) -> jnp.ndarray:
    """
    Current valid cache length.

    Returns
    ─────────────────────────────────────────
    scalar int32
    """

    return cache.cache_position
