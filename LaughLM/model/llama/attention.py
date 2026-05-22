"""
LaughLM/model/llama/attention.py

Canonical Llama attention.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from flax import linen as nn

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.initialization import (
    create_dense,
    constrain_hidden_states,
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

from LaughLM.runtime.attention.backend import (
    apply_attention,
)

from LaughLM.runtime.attention.types import (
    AttentionBackend,
    AttentionMaskSpec,
    AttentionMaskType,
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

        # ====================================================
        # Projection layers
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
        # KV cache constraints
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
        # IMPORTANT
        #
        # attention_mask is intentionally ignored.
        #
        # Runtime masking is generated dynamically
        # via AttentionMaskSpec.
        #
        # This keeps:
        # - flash attention
        # - online softmax
        # - decode specialization
        #
        # backend-native.
        # ====================================================

        mask_spec = AttentionMaskSpec(
            mask_type=AttentionMaskType(
                config.attention_mask_type
            ),
            sliding_window=config.sliding_window,
            chunk_size=config.chunk_size,
        )

        backend = AttentionBackend(
            config.attention_backend
        )

        # ====================================================
        # Runtime attention
        # ====================================================

        attn_output = apply_attention(
            query_states,
            key_states,
            value_states,
            mask_spec=mask_spec,
            backend=backend,
            block_q=config.attention_block_q,
            block_kv=config.attention_block_kv,
        )

        # ====================================================
        # Restore hidden layout
        # ====================================================

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
