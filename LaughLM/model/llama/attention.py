"""
LaughLM/model/llama/attention.py

Canonical Llama attention.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from flax import linen as nn

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.initialization import (
    create_dense,
    constrain_hidden_states,
    constrain_attention_q,
    constrain_attention_kv,
)

from LaughLM.model.llama.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from LaughLM.model.llama.kv_cache import (
    KVCache,
    update_kv_cache,
)

from LaughLM.distributed.sharding import (
    constrain_kv_cache,
)


# ============================================================
# GQA helper
# ============================================================

def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    """
    Expand KV heads for GQA.

    Input:
        [B, KVH, T, Dh]

    Output:
        [B, QH, T, Dh]
    """

    if n_rep == 1:
        return hidden_states

    b, kvh, t, dh = (
        hidden_states.shape
    )

    hidden_states = hidden_states[
        :,
        :,
        None,
        :,
        :,
    ]

    hidden_states = jnp.broadcast_to(
        hidden_states,
        (
            b,
            kvh,
            n_rep,
            t,
            dh,
        ),
    )

    return hidden_states.reshape(
        b,
        kvh * n_rep,
        t,
        dh,
    )


# ============================================================
# Attention
# ============================================================

class LlamaAttention(nn.Module):

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        positions: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        mode: str = "train",
    ) -> tuple[
        jnp.ndarray,
        Optional[KVCache],
    ]:

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        config = self.config

        B, T, _ = hidden_states.shape

        num_heads = (
            config.num_attention_heads
        )

        num_kv_heads = (
            config.num_key_value_heads
        )

        head_dim = config.head_dim

        num_kv_groups = (
            num_heads // num_kv_heads
        )

        # ====================================================
        # Projections
        # ====================================================

        q_proj = create_dense(
            features=num_heads * head_dim,
            config=config,
            use_bias=config.attention_bias,
            name="q_proj",
        )

        k_proj = create_dense(
            features=num_kv_heads * head_dim,
            config=config,
            use_bias=config.attention_bias,
            name="k_proj",
        )

        v_proj = create_dense(
            features=num_kv_heads * head_dim,
            config=config,
            use_bias=config.attention_bias,
            name="v_proj",
        )

        o_proj = create_dense(
            features=config.hidden_size,
            config=config,
            use_bias=config.attention_bias,
            name="o_proj",
        )

        # ====================================================
        # QKV projections
        # ====================================================

        query_states = q_proj(
            hidden_states
        )

        key_states = k_proj(
            hidden_states
        )

        value_states = v_proj(
            hidden_states
        )

        # ====================================================
        # Reshape
        # ====================================================

        query_states = query_states.reshape(
            B,
            T,
            num_heads,
            head_dim,
        )

        key_states = key_states.reshape(
            B,
            T,
            num_kv_heads,
            head_dim,
        )

        value_states = value_states.reshape(
            B,
            T,
            num_kv_heads,
            head_dim,
        )

        # ====================================================
        # RoPE
        # ====================================================

        rotary_emb = RotaryEmbedding(
            config
        )

        cos, sin = rotary_emb(
            query_states,
            positions,
        )

        (
            query_states,
            key_states,
        ) = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        # ====================================================
        # KV cache layout constraints
        #
        # [B, S, KVH, Dh]
        # ====================================================

        key_states = constrain_kv_cache(
            key_states
        )

        value_states = constrain_kv_cache(
            value_states
        )

        # ====================================================
        # KV cache update
        # ====================================================

        updated_cache = None

        if kv_cache is not None:

            (
                updated_cache,
                key_states,
                value_states,
            ) = update_kv_cache(
                kv_cache,
                key_states,
                value_states,
            )

            kv_length = (
                updated_cache
                .cache_position
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # Slice BEFORE transpose.
            #
            # key/value layout currently:
            #   [B, S, KVH, Dh]
            # ------------------------------------------------

            key_states = key_states[
                :,
                :kv_length,
                :,
                :,
            ]

            value_states = value_states[
                :,
                :kv_length,
                :,
                :,
            ]

        # ====================================================
        # Attention transpose
        # ====================================================

        query_states = jnp.transpose(
            query_states,
            (0, 2, 1, 3),
        )

        key_states = jnp.transpose(
            key_states,
            (0, 2, 1, 3),
        )

        value_states = jnp.transpose(
            value_states,
            (0, 2, 1, 3),
        )

        # ====================================================
        # Logical constraints
        # ====================================================

        query_states = constrain_attention_q(
            query_states
        )

        key_states = constrain_attention_kv(
            key_states
        )

        value_states = constrain_attention_kv(
            value_states
        )

        # ====================================================
        # GQA expansion
        # ====================================================

        key_states = repeat_kv(
            key_states,
            num_kv_groups,
        )

        value_states = repeat_kv(
            value_states,
            num_kv_groups,
        )

        # ====================================================
        # Attention logits
        # ====================================================

        attn_weights = jnp.matmul(
            query_states,
            jnp.swapaxes(
                key_states,
                -1,
                -2,
            ),
            preferred_element_type=jnp.float32,
        )

        attn_weights = (
            attn_weights
            * (head_dim ** -0.5)
        )

        # ====================================================
        # Mask
        # ====================================================

        if attention_mask is not None:

            attn_weights = (
                attn_weights
                + attention_mask
            )

        # ====================================================
        # Stable fp32 softmax
        # ====================================================

        attn_weights = (
            attn_weights.astype(
                jnp.float32
            )
        )

        attn_weights = jax.nn.softmax(
            attn_weights,
            axis=-1,
        )

        attn_weights = (
            attn_weights.astype(
                query_states.dtype
            )
        )

        # ====================================================
        # Attention output
        # ====================================================

        attn_output = jnp.matmul(
            attn_weights,
            value_states,
            preferred_element_type=jnp.float32,
        )

        # ====================================================
        # Restore hidden-state layout
        # ====================================================

        attn_output = jnp.transpose(
            attn_output,
            (0, 2, 1, 3),
        )

        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        # ====================================================
        # Output projection
        # ====================================================

        attn_output = o_proj(
            attn_output
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        return (
            attn_output,
            updated_cache,
        )
