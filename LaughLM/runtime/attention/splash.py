"""
LaughLM/runtime/attention/splash.py

Real TPU SplashAttention runtime.

Uses:
    jax.experimental.pallas.ops.tpu.splash_attention

This provides:
    - tiled streaming attention
    - O(T) memory
    - TPU fused kernels
    - frontier-scale sequence support
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)

from .types import (
    AttentionMaskSpec,
)

Array = jnp.ndarray


# ============================================================
# Block helpers
# ============================================================

def _find_block_size(
    seq_len: int,
) -> tuple[int, int]:
    """
    Find TPU-friendly block size.

    Returns:
        block_size,
        padding_needed
    """

    for block in (
        512,
        256,
        128,
    ):

        if seq_len % block == 0:
            return block, 0

    block = 512

    padded = (
        math.ceil(seq_len / block)
        * block
    )

    return (
        block,
        padded - seq_len,
    )


# ============================================================
# Padding
# ============================================================

def _pad_attention_tensors(
    query,
    key,
    value,
):

    B, Tq, Hq, D = query.shape
    _, Tk, Hkv, _ = key.shape

    assert Tq == Tk

    block_size, pad = _find_block_size(
        Tq
    )

    if pad == 0:

        return (
            query,
            key,
            value,
            block_size,
            0,
        )

    query = jnp.pad(
        query,
        (
            (0, 0),
            (0, pad),
            (0, 0),
            (0, 0),
        ),
    )

    key = jnp.pad(
        key,
        (
            (0, 0),
            (0, pad),
            (0, 0),
            (0, 0),
        ),
    )

    value = jnp.pad(
        value,
        (
            (0, 0),
            (0, pad),
            (0, 0),
            (0, 0),
        ),
    )

    return (
        query,
        key,
        value,
        block_size,
        pad,
    )


# ============================================================
# SplashAttention
# ============================================================

def splash_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
):
    """
    TPU SplashAttention.

    query:
        [B, T, Hq, D]

    key/value:
        [B, T, Hkv, D]
    """

    del mask_spec

    B, T, Hq, D = query.shape
    _, _, Hkv, _ = key.shape

    #
    # SplashAttention supports GQA natively.
    #
    # We keep:
    #
    # query : [B, T, Hq, D]
    # key   : [B, T, Hkv, D]
    #

    query, key, value, block_size, pad = (
        _pad_attention_tensors(
            query,
            key,
            value,
        )
    )

    T_padded = query.shape[1]

    # ========================================================
    # Splash expects BNTH
    # ========================================================

    query = jnp.transpose(
        query,
        (
            0,
            2,
            1,
            3,
        ),
    )

    key = jnp.transpose(
        key,
        (
            0,
            2,
            1,
            3,
        ),
    )

    value = jnp.transpose(
        value,
        (
            0,
            2,
            1,
            3,
        ),
    )

    # ========================================================
    # Causal mask
    # ========================================================

    causal_mask = (
        splash_attention_mask.CausalMask(
            shape=(
                T_padded,
                T_padded,
            ),
        )
    )

    multi_head_mask = (
        splash_attention_mask.MultiHeadMask(
            masks=(
                causal_mask,
            ) * Hq,
        )
    )

    # ========================================================
    # Kernel blocks
    # ========================================================

    block_sizes = (
        splash_attention_kernel.BlockSizes(
            block_q=block_size,
            block_kv=block_size,
            block_kv_compute=block_size,

            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_kv_dkv_compute=block_size,

            block_q_dq=block_size,
            block_kv_dq=block_size,
        )
    )

    kernel = (
        splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            block_sizes=block_sizes,
            head_shards=1,
            q_seq_shards=1,
        )
    )

    # ========================================================
    # Batch vmap
    # ========================================================

    output = jax.vmap(
        kernel,
        in_axes=(
            0,
            0,
            0,
            None,
        ),
    )(
        query,
        key,
        value,
        None,
    )

    # ========================================================
    # Restore layout
    # ========================================================

    output = jnp.transpose(
        output,
        (
            0,
            2,
            1,
            3,
        ),
    )

    if pad > 0:

        output = output[
            :,
            :-pad,
            :,
            :,
        ]

    return output
