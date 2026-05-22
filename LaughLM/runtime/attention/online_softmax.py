"""
LaughLM/runtime/attention/online_softmax.py

WARNING:
    THIS IS A RESEARCH / DEBUG IMPLEMENTATION.

    This file implements online softmax attention
    directly in JAX primitives.

    It is NOT production efficient on TPU.

    Large sequence training should use:
        jax.nn.dot_product_attention()

    which dispatches to fused kernels.

WHY THIS EXISTS:
    - correctness testing
    - experimentation
    - algorithm research
    - reference implementation

NOT FOR:
    - production TPU training
    - large sequence compile
    - frontier scaling
"""

from __future__ import annotations

import math

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

    RESEARCH IMPLEMENTATION ONLY.
    """

    B, T, Hq, D = query.shape
    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    scale = D ** -0.5

    query_dtype = query.dtype

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

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
            T,
            Hkv,
            groups,
            1,
        ),
        -jnp.inf,
        dtype=jnp.float32,
    )

    l_i = jnp.zeros(
        (
            B,
            T,
            Hkv,
            groups,
            1,
        ),
        dtype=jnp.float32,
    )

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

        kv_len = kv_end - kv_start

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

        logits = jnp.einsum(
            "bthgd,bshd->bthgs",
            query.astype(jnp.float32),
            k_block.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )

        logits = logits * scale

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
        )[
            None,
            :,
            None,
            None,
            :
        ]

        block_m = jnp.max(
            logits,
            axis=-1,
            keepdims=True,
        )

        new_m = jnp.maximum(
            m_i,
            block_m,
        )

        exp_old = jnp.exp(
            m_i - new_m
        )

        exp_block = jnp.exp(
            logits - new_m
        )

        block_l = jnp.sum(
            exp_block,
            axis=-1,
            keepdims=True,
        )

        new_l = (
            exp_old * l_i
            + block_l
        )

        old_scale = (
            exp_old * l_i
        ) / jnp.maximum(
            new_l,
            1e-6,
        )

        output = (
            output
            * old_scale
        )

        block_out = jnp.einsum(
            "bthgs,bshd->bthgd",
            exp_block.astype(
                v_block.dtype
            ),
            v_block,
            preferred_element_type=jnp.float32,
        )

        output = output + (
            block_out
            / jnp.maximum(
                new_l,
                1e-6,
            )
        )

        m_i = new_m
        l_i = new_l

    output = output.reshape(
        B,
        T,
        Hq,
        D,
    )

    return output.astype(
        query_dtype
    )
