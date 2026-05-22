"""
LaughLM/runtime/attention/decode.py
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


def decode_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
    *,
    block_kv: int = 128,
) -> Array:
    """
    Specialized autoregressive decode attention.

    Optimized for:
        query length = 1

    query:
        [B, 1, Hq, D]

    key/value:
        [B, S, Hkv, D]
    """

    B, T, Hq, D = query.shape

    assert T == 1

    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    # =====================================================
    # Reshape query into GQA groups
    #
    # [B, 1, Hq, D]
    # ->
    # [B, 1, Hkv, G, D]
    # =====================================================

    query = query.reshape(
        B,
        1,
        Hkv,
        groups,
        D,
    )

    # =====================================================
    # Running online softmax state
    # =====================================================

    running_max = jnp.full(
        (
            B,
            Hkv,
            groups,
            1,
        ),
        -jnp.inf,
        dtype=jnp.float32,
    )

    running_sum = jnp.zeros(
        (
            B,
            Hkv,
            groups,
            1,
        ),
        dtype=jnp.float32,
    )

    running_out = jnp.zeros(
        (
            B,
            1,
            Hkv,
            groups,
            D,
        ),
        dtype=jnp.float32,
    )

    # =====================================================
    # Build mask once
    # =====================================================

    full_mask = build_mask(
        q_len=1,
        kv_len=S,
        spec=mask_spec,
    )

    full_mask = full_mask[
        None,
        None,
        None,
        :,
        :,
    ]

    scale = D ** -0.5

    # =====================================================
    # KV block traversal
    # =====================================================

    num_kv_blocks = (
        S + block_kv - 1
    ) // block_kv

    def kv_loop(
        kv_block_idx,
        state,
    ):

        (
            running_max,
            running_sum,
            running_out,
        ) = state

        kv_start = (
            kv_block_idx * block_kv
        )

        kv_size = min(
            block_kv,
            S - kv_start,
        )

        # -------------------------------------------------
        # KV slices
        # -------------------------------------------------

        k_block = jax.lax.dynamic_slice(
            key,
            (
                0,
                kv_start,
                0,
                0,
            ),
            (
                B,
                kv_size,
                Hkv,
                D,
            ),
        )

        v_block = jax.lax.dynamic_slice(
            value,
            (
                0,
                kv_start,
                0,
                0,
            ),
            (
                B,
                kv_size,
                Hkv,
                D,
            ),
        )

        # -------------------------------------------------
        # Mask slice
        # -------------------------------------------------

        mask_block = jax.lax.dynamic_slice(
            full_mask,
            (
                0,
                0,
                0,
                0,
                kv_start,
            ),
            (
                1,
                1,
                1,
                1,
                kv_size,
            ),
        )

        # -------------------------------------------------
        # QK
        #
        # logits:
        # [B, Hkv, G, 1, K]
        # -------------------------------------------------

        logits = jnp.einsum(
            "bthgd,bshd->bhgts",
            query.astype(jnp.float32),
            k_block.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )

        logits = logits * scale

        logits = logits + jnp.where(
            mask_block,
            0.0,
            DEFAULT_MASK_VALUE,
        )

        # -------------------------------------------------
        # Online softmax update
        # -------------------------------------------------

        block_max = jnp.max(
            logits,
            axis=-1,
        )

        new_max = jnp.maximum(
            running_max,
            block_max,
        )

        old_scale = jnp.exp(
            running_max - new_max
        )

        block_scale = jnp.exp(
            block_max - new_max
        )

        probs = jnp.exp(
            logits
            - new_max[..., None]
        )

        block_sum = jnp.sum(
            probs,
            axis=-1,
        )

        running_sum = (
            running_sum * old_scale
            + block_sum
        )

        block_out = jnp.einsum(
            "bhgts,bshd->bthgd",
            probs,
            v_block.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )

        running_out = (
            running_out
            * old_scale[
                :,
                :,
                :,
                :,
                None,
            ]
            + block_out
        )

        running_max = new_max

        return (
            running_max,
            running_sum,
            running_out,
        )

    (
        running_max,
        running_sum,
        running_out,
    ) = jax.lax.fori_loop(
        0,
        num_kv_blocks,
        kv_loop,
        (
            running_max,
            running_sum,
            running_out,
        ),
    )

    # =====================================================
    # Final normalize
    # =====================================================

    out = (
        running_out
        / running_sum[
            :,
            :,
            :,
            :,
            None,
        ]
    )

    # =====================================================
    # Restore layout
    # =====================================================

    out = out.reshape(
        B,
        1,
        Hq,
        D,
    )

    return out.astype(query.dtype)
