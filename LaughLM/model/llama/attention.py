"""
LaughLM/model/llama/attention.py

Canonical Llama attention.

Design goals:
- HF-compatible semantics
- deterministic KV-cache behavior
- explicit prefill/decode modes
- stable GQA implementation
- minimal architecture surface

Tensor conventions
------------------
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

from flax import linen as nn

import jax
import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig
from LaughLM.model.llama.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from LaughLM.model.llama.kv_cache import (
    KVCache,
    update_kv_cache,
)


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

        B, T, _ = hidden_states.shape

        config = self.config

        num_heads = config.num_attention_heads

        num_kv_heads = config.num_key_value_heads

        head_dim = config.head_dim

        num_kv_groups = (
            num_heads // num_kv_heads
        )

        # ──────────────────────────────────────────
        # Projections
        # ──────────────────────────────────────────

        q_proj = nn.Dense(
            num_heads * head_dim,
            use_bias=config.attention_bias,
            name="q_proj",
        )

        k_proj = nn.Dense(
            num_kv_heads * head_dim,
            use_bias=config.attention_bias,
            name="k_proj",
        )

        v_proj = nn.Dense(
            num_kv_heads * head_dim,
            use_bias=config.attention_bias,
            name="v_proj",
        )

        o_proj = nn.Dense(
            config.hidden_size,
            use_bias=config.attention_bias,
            name="o_proj",
        )

        query_states = q_proj(hidden_states)

        key_states = k_proj(hidden_states)

        value_states = v_proj(hidden_states)

        # ──────────────────────────────────────────
        # Reshape
        # ──────────────────────────────────────────

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

        # ──────────────────────────────────────────
        # RoPE
        # ──────────────────────────────────────────

        rotary_emb = RotaryEmbedding(config)

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

        # ──────────────────────────────────────────
        # KV cache update
        # ──────────────────────────────────────────

        updated_cache = None

        if kv_cache is not None:

            updated_cache, key_states, value_states = (
                update_kv_cache(
                    kv_cache,
                    key_states,
                    value_states,
                )
            )

        # ──────────────────────────────────────────
        # Transpose for attention
        # ──────────────────────────────────────────

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

        # ──────────────────────────────────────────
        # GQA expansion
        # ──────────────────────────────────────────

        key_states = repeat_kv(
            key_states,
            num_kv_groups,
        )

        value_states = repeat_kv(
            value_states,
            num_kv_groups,
        )

        # ──────────────────────────────────────────
        # Attention
        # ──────────────────────────────────────────

        attn_weights = jnp.matmul(
            query_states,
            jnp.swapaxes(key_states, -1, -2),
        )

        attn_weights = (
            attn_weights
            * (head_dim ** -0.5)
        )

        if attention_mask is not None:
            attn_weights = (
                attn_weights
                + attention_mask
            )

        attn_weights = jax.nn.softmax(
            attn_weights.astype(jnp.float32),
            axis=-1,
        ).astype(query_states.dtype)

        attn_output = jnp.matmul(
            attn_weights,
            value_states,
        )

        # ──────────────────────────────────────────
        # Output reshape
        # ──────────────────────────────────────────

        attn_output = jnp.transpose(
            attn_output,
            (0, 2, 1, 3),
        )

        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        attn_output = o_proj(attn_output)

        return attn_output, updated_cache