"""
LaughLM/runtime/attention/online_softmax.py

Streaming FlashAttention-style online softmax.

Memory efficient:
- no full attention matrix
- no full mask materialization
- block streaming KV
- GQA-native
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .block_mask import (
    make_block_mask,
)

from .types import (
    AttentionMaskSpec,
    DEFAULT_MASK_VALUE,
)

Array = jnp.ndarray


def online_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
    *,
    block_q: int = 128,
    block_kv: int = 128,
) -> Array:
    """
    Streaming online softmax attention.

    query:
        [B, T, Hq, D]

    key/value:
        [B, S, Hkv, D]
    """

    B, T, Hq, D = query.shape
    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    scale = D ** -0.5

    # ======================================================
    # Reshape query for GQA
    # ======================================================

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

    # ======================================================
    # Output buffers
    # ======================================================

    output = jnp.zeros(
        (
            B,
            T,
            Hkv,
            groups,
            D,
        ),
        dtype=jnp.float32,
    )

    m_i = jnp.full(
        (
            B,
            Hkv,
            groups,
            T,
        ),
        -jnp.inf,
        dtype=jnp.float32,
    )

    l_i = jnp.zeros(
        (
            B,
            Hkv,
            groups,
            T,
        ),
        dtype=jnp.float32,
    )

    # ======================================================
    # KV streaming loop
    # ======================================================

    num_kv_blocks = math.ceil(
        S / block_kv
    )

    for kv_block_idx in range(
        num_kv_blocks
    ):

        kv_start = (
            kv_block_idx * block_kv
        )

        kv_end = min(
            kv_start + block_kv,
            S,
        )

        k_block = key[
            :,
            kv_start:kv_end,
            :,
            :,
        ]

        v_block = value[
            :,
            kv_start:kv_end,
            :,
            :,
        ]

        kv_len = kv_end - kv_start

        # ==================================================
        # Block logits
        # ==================================================

        logits = jnp.einsum(
            "bthgd,bshd->bhgts",
            query.astype(jnp.float32),
            k_block.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )

        logits = logits * scale

        # ==================================================
        # Block-local mask
        # ==================================================

        mask = make_block_mask(
            q_start=0,
            q_len=T,
            kv_start=kv_start,
            kv_len=kv_len,
            spec=mask_spec,
        )

        logits = logits + jnp.where(
            mask,
            0.0,
            DEFAULT_MASK_VALUE,
        )[None, None, None, :, :]

        # ==================================================
        # Online softmax
        # ==================================================

        block_m = jnp.max(
            logits,
            axis=-1,
        )

        new_m = jnp.maximum(
            m_i,
            block_m,
        )

        exp_old = jnp.exp(
            m_i - new_m
        )

        exp_block = jnp.exp(
            logits
            - new_m[..., None]
        )

        block_l = jnp.sum(
            exp_block,
            axis=-1,
        )

        new_l = (
            exp_old * l_i
            + block_l
        )

        # ==================================================
        # Rescale output accumulator
        # ==================================================

        output = (
            output
            * (
                (
                    exp_old * l_i
                )
                / jnp.maximum(
                    new_l,
                    1e-6,
                )
            )[..., None]
        )

        block_out = jnp.einsum(
            "bhgts,bshd->bthgd",
            exp_block.astype(v_block.dtype),
            v_block,
            preferred_element_type=jnp.float32,
        )

        output = output + (
            block_out
            / jnp.maximum(
                new_l,
                1e-6,
            )[..., None]
        )

        m_i = new_m
        l_i = new_l

    # ======================================================
    # Restore head layout
    # ======================================================

    output = output.reshape(
        B,
        T,
        Hq,
        D,
    )

    return output.astype(query.dtype)
