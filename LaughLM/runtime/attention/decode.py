"""
LaughLM/runtime/attention/decode.py
"""

from __future__ import annotations

from .backend import apply_attention
from .kv_cache import (
    append_kv_cache,
    get_kv_cache_view,
)


def decode_step(
    *,
    query,
    key,
    value,
    cache,
    mask_spec,
    backend,
):
    """
    Single autoregressive decode step.

    query:
        [B, 1, Hq, D]

    key/value:
        [B, 1, Hkv, D]
    """

    #
    # Append new KV
    #

    cache = append_kv_cache(
        cache,
        key,
        value,
    )

    #
    # Fetch active cache
    #

    cached_key, cached_value, _ = (
        get_kv_cache_view(cache)
    )

    #
    # Attention over full cache
    #

    out = apply_attention(
        query,
        cached_key,
        cached_value,
        mask_spec,
        backend=backend,
    )

    return out, cache
