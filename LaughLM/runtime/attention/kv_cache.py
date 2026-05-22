LaughLM/runtime/attention/kv_cache.py

Minimal KV cache runtime for:
- prefill
- autoregressive decode

Tensor layout:
    key/value:
        [B, S, Hkv, D]
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jnp.ndarray


@dataclass
class KVCache:
    """
    Runtime KV cache.

    Attributes
    ----------
    key:
        [B, max_seq_len, Hkv, D]

    value:
        [B, max_seq_len, Hkv, D]

    lengths:
        [B]
    """

    key: Array
    value: Array
    lengths: Array


def init_kv_cache(
    *,
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=jnp.bfloat16,
) -> KVCache:
    """
    Initialize empty KV cache.
    """

    key = jnp.zeros(
        (
            batch_size,
            max_seq_len,
            num_kv_heads,
            head_dim,
        ),
        dtype=dtype,
    )

    value = jnp.zeros_like(key)

    lengths = jnp.zeros(
        (batch_size,),
        dtype=jnp.int32,
    )

    return KVCache(
        key=key,
        value=value,
        lengths=lengths,
    )


def prefill_kv_cache(
    cache: KVCache,
    key: Array,
    value: Array,
) -> KVCache:
    """
    Fill KV cache from prompt.

    key/value:
        [B, S, Hkv, D]
    """

    B, S, Hkv, D = key.shape

    cache_key = jax.lax.dynamic_update_slice(
        cache.key,
        key,
        (
            0,
            0,
            0,
            0,
        ),
    )

    cache_value = jax.lax.dynamic_update_slice(
        cache.value,
        value,
        (
            0,
            0,
            0,
            0,
        ),
    )

    lengths = jnp.full(
        (B,),
        S,
        dtype=jnp.int32,
    )

    return KVCache(
        key=cache_key,
        value=cache_value,
        lengths=lengths,
    )


def append_kv_cache(
    cache: KVCache,
    key: Array,
    value: Array,
) -> KVCache:
    """
    Append one decode token.

    key/value:
        [B, 1, Hkv, D]
    """

    B = key.shape[0]

    #
    # Per-batch insertion positions
    #

    positions = cache.lengths

    def update_single_batch(
        cache_key,
        cache_value,
        key_i,
        value_i,
        pos,
    ):

        cache_key = jax.lax.dynamic_update_slice(
            cache_key,
            key_i,
            (
                pos,
                0,
                0,
            ),
        )

        cache_value = jax.lax.dynamic_update_slice(
            cache_value,
            value_i,
            (
                pos,
                0,
                0,
            ),
        )

        return cache_key, cache_value

    cache_key, cache_value = jax.vmap(
        update_single_batch,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
        ),
    )(
        cache.key,
        cache.value,
        key,
        value,
        positions,
    )

    return KVCache(
        key=cache_key,
        value=cache_value,
        lengths=cache.lengths + 1,
    )


def get_kv_cache_view(
    cache: KVCache,
):
    """
    Return active KV tensors.

    Returns
    -------
    key:
        [B, S_active, Hkv, D]

    value:
        [B, S_active, Hkv, D]
    """

    #
    # Current implementation assumes
    # equal decode lengths across batch.
    #

    max_len = jnp.max(cache.lengths)

    key = jax.lax.dynamic_slice(
        cache.key,
        (
            0,
            0,
            0,
            0,
        ),
        (
            cache.key.shape[0],
            max_len,
            cache.key.shape[2],
            cache.key.shape[3],
        ),
    )

    value = jax.lax.dynamic_slice(
        cache.value,
        (
            0,
            0,
            0,
            0,
        ),
        (
            cache.value.shape[0],
            max_len,
            cache.value.shape[2],
            cache.value.shape[3],
        ),
    )

    return (
        key,
        value,
        cache.lengths,
    )
