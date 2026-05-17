"""
LaughLM/model/llama/attention.py

Canonical Llama attention.

Frontier-grade SPMD additions:
────────────────────────────────────────────────────
1. Logical activation constraints
2. Tensor-parallel aware Q/K/V sharding
3. KV-head-aware GQA partitioning
4. Mesh-safe attention layouts
5. Stable bf16 softmax path
6. Sequence-parallel ready tensor semantics
7. GSPMD-compatible attention pipeline

Design goals:
- HF-compatible semantics
- deterministic KV-cache behavior
- explicit prefill/decode modes
- stable GQA implementation
- minimal architecture surface
- deterministic initialization semantics
- tensor-parallel compatible layouts

Tensor conventions
──────────────────
Input hidden states:
    [B, T, D]

Q:
    [B, QH, T, Dh]

K/V:
    [B, KVH, T, Dh]

Attention output:
    [B, T, D]

Cache storage:
    [B, S, KVH, Dh]
"""

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


# ─────────────────────────────────────────────────────────────
# GQA helper
# ─────────────────────────────────────────────────────────────

def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    """
    Repeat KV heads for grouped-query attention.

    Input:
        [B, KVH, T, Dh]

    Output:
        [B, QH, T, Dh]
    """

    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = (
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
            batch,
            num_kv_heads,
            n_rep,
            seq_len,
            head_dim,
        ),
    )

    return hidden_states.reshape(
        batch,
        num_kv_heads * n_rep,
        seq_len,
        head_dim,
    )


# ─────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────

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
        """
        Parameters
        ----------
        hidden_states:
            [B, T, D]

        positions:
            [B, T]

        attention_mask:
            [B, 1, T_q, T_kv]

        mode:
            "train"
            "prefill"
            "decode"
        """

        # --------------------------------------------------
        # Hidden-state constraint
        # --------------------------------------------------

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        B, T, _ = hidden_states.shape

        config = self.config

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

        # --------------------------------------------------
        # Projections
        # --------------------------------------------------

        q_proj = create_dense(
            features=(
                num_heads * head_dim
            ),
            config=config,
            use_bias=config.attention_bias,
            name="q_proj",
        )

        k_proj = create_dense(
            features=(
                num_kv_heads * head_dim
            ),
            config=config,
            use_bias=config.attention_bias,
            name="k_proj",
        )

        v_proj = create_dense(
            features=(
                num_kv_heads * head_dim
            ),
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

        # --------------------------------------------------
        # QKV projections
        # --------------------------------------------------

        query_states = q_proj(
            hidden_states
        )

        key_states = k_proj(
            hidden_states
        )

        value_states = v_proj(
            hidden_states
        )

        # --------------------------------------------------
        # Reshape
        # --------------------------------------------------

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

        # --------------------------------------------------
        # RoPE
        # --------------------------------------------------

        rotary_emb = RotaryEmbedding(
            config
        )

        cos, sin = rotary_emb(
            query_states,
            positions,
        )

        query_states, key_states = (
            apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
        )

        # --------------------------------------------------
        # KV cache update
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Transpose for attention
        # --------------------------------------------------
        #
        # Q:
        #   [B, Tq, QH, Dh]
        #       ->
        #   [B, QH, Tq, Dh]
        #
        # K/V:
        #   [B, Tk, KVH, Dh]
        #       ->
        #   [B, KVH, Tk, Dh]
        #

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

        # --------------------------------------------------
        # Logical constraints
        # --------------------------------------------------

        query_states = constrain_attention_q(
            query_states
        )

        key_states = constrain_attention_kv(
            key_states
        )

        value_states = constrain_attention_kv(
            value_states
        )

        # --------------------------------------------------
        # GQA expansion
        # --------------------------------------------------

        key_states = repeat_kv(
            key_states,
            num_kv_groups,
        )

        value_states = repeat_kv(
            value_states,
            num_kv_groups,
        )

        # --------------------------------------------------
        # Attention logits
        # --------------------------------------------------

        attn_weights = jnp.matmul(
            query_states,
            jnp.swapaxes(
                key_states,
                -1,
                -2,
            ),
        )

        attn_weights = (
            attn_weights
            * (head_dim ** -0.5)
        )

        # --------------------------------------------------
        # Attention mask
        # --------------------------------------------------

        if attention_mask is not None:

            attn_weights = (
                attn_weights
                + attention_mask
            )

        # --------------------------------------------------
        # Numerically-stable softmax
        # --------------------------------------------------
        #
        # Always compute softmax in fp32.
        #
        # Frontier standard:
        #   bf16 logits
        #       ->
        #   fp32 softmax
        #       ->
        #   cast back
        #

        attn_weights = (
            jax.nn.softmax(
                attn_weights.astype(
                    jnp.float32
                ),
                axis=-1,
            ).astype(
                query_states.dtype
            )
        )

        # --------------------------------------------------
        # Attention output
        # --------------------------------------------------

        attn_output = jnp.matmul(
            attn_weights,
            value_states,
        )

        # --------------------------------------------------
        # Output reshape
        # --------------------------------------------------

        attn_output = jnp.transpose(
            attn_output,
            (0, 2, 1, 3),
        )

        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        # --------------------------------------------------
        # Hidden-state constraint
        # --------------------------------------------------

        attn_output = constrain_hidden_states(
            attn_output
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        attn_output = o_proj(
            attn_output
        )

        # --------------------------------------------------
        # Final activation constraint
        # --------------------------------------------------

        attn_output = constrain_hidden_states(
            attn_output
        )

        return (
            attn_output,
            updated_cache,
        )
