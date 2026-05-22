"""
LaughLM/runtime/attention/flash.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .masking import build_mask
from .online_softmax import (
    online_softmax_update,
)
from .types import (
    AttentionMaskSpec,
    DEFAULT_MASK_VALUE,
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
):
    """
    FlashAttention v1.

    Memory efficient attention.

    query:
        [B, T, Hq, D]

    key/value:
        [B, S, Hkv, D]
    """

    B, T, Hq, D = query.shape
    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    #
    # Reshape query into GQA groups
    #

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

    #
    # Output tensor
    #

    output = jnp.zeros(
        (
            B,
            T,
            Hkv,
            groups,
            D,
        ),
        dtype=query.dtype,
    )

    #
    # Number of blocks
    #

    num_q_blocks = (
        T + block_q - 1
    ) // block_q

    num_kv_blocks = (
        S + block_kv - 1
    ) // block_kv

    #
    # Full mask
    #

    full_mask = build_mask(
        q_len=T,
        kv_len=S,
        spec=mask_spec,
    )

    #
    # Q block loop
    #

    def q_loop(
        q_block_idx,
        output,
    ):

        q_start = q_block_idx * block_q

        q_size = min(
            block_q,
            T - q_start,
        )

        #
        # Q block
        #

        q_block = jax.lax.dynamic_slice(
            query,
            (
                0,
                q_start,
                0,
                0,
                0,
            ),
            (
                B,
                q_size,
                Hkv,
                groups,
                D,
            ),
        )

        #
        # Online softmax state
        #

        running_max = jnp.full(
            (
                B,
                Hkv,
                groups,
                q_size,
            ),
            -jnp.inf,
            dtype=jnp.float32,
        )

        running_sum = jnp.zeros(
            (
                B,
                Hkv,
                groups,
                q_size,
            ),
            dtype=jnp.float32,
        )

        running_out = jnp.zeros(
            (
                B,
                Hkv,
                groups,
                q_size,
                D,
            ),
            dtype=jnp.float32,
        )

        #
        # KV block loop
        #

        def kv_loop(
            kv_block_idx,
            state,
        ):

            (
                running_max,
                running_sum,
                running_out,
            ) = state

            kv_start = kv_block_idx * block_kv

            kv_size = min(
                block_kv,
                S - kv_start,
            )

            #
            # KV slices
            #

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

            #
            # QK
            #

            logits = jnp.einsum(
                "bthgd,bshd->bhgts",
                q_block.astype(jnp.float32),
                k_block.astype(jnp.float32),
            )

            #
            # Local mask slice
            #

            mask_block = jax.lax.dynamic_slice(
                full_mask,
                (
                    q_start,
                    kv_start,
                ),
                (
                    q_size,
                    kv_size,
                ),
            )

            logits = logits + jnp.where(
                mask_block,
                0.0,
                DEFAULT_MASK_VALUE,
            )[None, None, None, :, :]

            #
            # Online update
            #

            (
                running_max,
                running_sum,
                running_out,
            ) = online_softmax_update(
                running_max,
                running_sum,
                running_out,
                logits,
                v_block.transpose(
                    0,
                    2,
                    1,
                    3,
                ),
            )

            return (
                running_max,
                running_sum,
                running_out,
            )

        (
            _,
            _,
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

        #
        # Restore layout
        #

        q_out = running_out.transpose(
            0,
            3,
            1,
            2,
            4,
        )

        #
        # Write output
        #

        output = jax.lax.dynamic_update_slice(
            output,
            q_out,
            (
                0,
                q_start,
                0,
                0,
                0,
            ),
        )

        return output

    output = jax.lax.fori_loop(
        0,
        num_q_blocks,
        q_loop,
        output,
    )

    #
    # Restore head layout
    #

    output = output.reshape(
        B,
        T,
        Hq,
        D,
    )

    return output
