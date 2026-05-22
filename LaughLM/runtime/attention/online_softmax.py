"""
LaughLM/runtime/attention/online_softmax.py
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


# ============================================================
# Streaming softmax update
# ============================================================

def online_softmax_update(
    prev_max: Array,
    prev_sum: Array,
    prev_out: Array,
    block_logits: Array,
    block_values: Array,
):
    """
    Streaming softmax update.

    prev_max:
        [..., T]

    prev_sum:
        [..., T]

    prev_out:
        [..., T, D]

    block_logits:
        [..., T, BK]

    block_values:
        [..., BK, D]
    """

    # --------------------------------------------------------
    # Local block max
    # --------------------------------------------------------

    block_max = jnp.max(
        block_logits,
        axis=-1,
    )

    # --------------------------------------------------------
    # Updated running max
    # --------------------------------------------------------

    new_max = jnp.maximum(
        prev_max,
        block_max,
    )

    # --------------------------------------------------------
    # Rescaling
    # --------------------------------------------------------

    prev_scale = jnp.exp(
        prev_max - new_max
    )

    block_scale = jnp.exp(
        block_max - new_max
    )

    # --------------------------------------------------------
    # Local exponentials
    # --------------------------------------------------------

    block_probs = jnp.exp(
        block_logits
        -
        block_max[..., None]
    )

    # --------------------------------------------------------
    # Local denominator
    # --------------------------------------------------------

    block_sum = jnp.sum(
        block_probs,
        axis=-1,
    )

    # --------------------------------------------------------
    # Updated denominator
    # --------------------------------------------------------

    new_sum = (
        prev_scale * prev_sum
        +
        block_scale * block_sum
    )

    # --------------------------------------------------------
    # Local output
    # --------------------------------------------------------

    block_out = jnp.einsum(
        "...tk,...kd->...td",
        block_probs,
        block_values,
    )

    # --------------------------------------------------------
    # Numerator accumulation
    # --------------------------------------------------------

    prev_term = (
        prev_out
        *
        prev_sum[..., None]
        *
        prev_scale[..., None]
    )

    block_term = (
        block_out
        *
        block_scale[..., None]
    )

    new_out = (
        prev_term
        +
        block_term
    ) / new_sum[..., None]

    return (
        new_max,
        new_sum,
        new_out,
    )


# ============================================================
# Generic tiled online attention
# ============================================================

def online_attention(
    query: Array,
    key: Array,
    value: Array,
    mask_spec: AttentionMaskSpec,
    *,
    block_q: int = 128,
    block_kv: int = 128,
):
    """
    Generic tiled online attention.

    This is NOT hardware flash attention.

    It is:
    - memory efficient
    - tiled
    - streaming softmax
    - backend portable

    query:
        [B, T, Hq, D]

    key/value:
        [B, S, Hkv, D]
    """

    B, T, Hq, D = query.shape

    _, S, Hkv, _ = key.shape

    assert Hq % Hkv == 0

    groups = Hq // Hkv

    # =====================================================
    # Reshape query into GQA groups
    # =====================================================

    query = query.reshape(
        B,
        T,
        Hkv,
        groups,
        D,
    )

    # =====================================================
    # Output tensor
    # =====================================================

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

    # =====================================================
    # Block counts
    # =====================================================

    num_q_blocks = (
        T + block_q - 1
    ) // block_q

    num_kv_blocks = (
        S + block_kv - 1
    ) // block_kv

    # =====================================================
    # Full mask
    # =====================================================

    full_mask = build_mask(
        q_len=T,
        kv_len=S,
        spec=mask_spec,
    )

    # =====================================================
    # Q block loop
    # =====================================================

    def q_loop(
        q_block_idx,
        output,
    ):

        q_start = (
            q_block_idx * block_q
        )

        q_size = min(
            block_q,
            T - q_start,
        )

        # -------------------------------------------------
        # Q block
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Running state
        # -------------------------------------------------

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

        # -------------------------------------------------
        # KV loop
        # -------------------------------------------------

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

            # ---------------------------------------------
            # KV slices
            # ---------------------------------------------

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

            # ---------------------------------------------
            # QK
            # ---------------------------------------------

            logits = jnp.einsum(
                "bthgd,bshd->bhgts",
                q_block.astype(jnp.float32),
                k_block.astype(jnp.float32),
            )

            logits *= (
                D ** -0.5
            )

            # ---------------------------------------------
            # Mask slice
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Online update
            # ---------------------------------------------

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

        # -------------------------------------------------
        # Restore layout
        # -------------------------------------------------

        q_out = running_out.transpose(
            0,
            3,
            1,
            2,
            4,
        )

        # -------------------------------------------------
        # Write output
        # -------------------------------------------------

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

    # =====================================================
    # Restore layout
    # =====================================================

    output = output.reshape(
        B,
        T,
        Hq,
        D,
    )

    return output.astype(query.dtype)
