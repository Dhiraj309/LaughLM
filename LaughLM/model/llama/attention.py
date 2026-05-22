"""
LaughLM/model/llama/attention.py

Canonical LLaMA attention for PMAP production.

Uses jax.nn.dot_product_attention instead of manual
matmul-softmax-matmul.

Current backend policy:
- standard / xla: official XLA dot_product_attention
- flash / cudnn: try cuDNN dot_product_attention on GPU, fallback to XLA
- decode: XLA dot_product_attention
- no Splash/Pallas here yet
"""

from __future__ import annotations

from typing import Optional
import warnings

import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.model.llama.config import LlamaConfig
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
from LaughLM.distributed.sharding import constrain_kv_cache


_LOGGED_BACKENDS = set()


def _log_attention_backend(name: str):
    if name not in _LOGGED_BACKENDS:
        _LOGGED_BACKENDS.add(name)
        print(f"[attention] using {name}", flush=True)


def _attention_impl_from_config(config: LlamaConfig, mode: str) -> str | None:
    """
    Returns requested JAX dot_product_attention implementation.

    None means JAX default/XLA path.
    """

    impl = getattr(config, "attention_impl", None)

    # Current LlamaConfig may not carry attention_impl yet.
    if impl is None:
        return None

    if mode == "decode":
        return None

    if impl in ("standard", "xla", "memory_efficient"):
        return None

    if impl in ("flash", "cudnn"):
        return "cudnn"

    return None


def _dot_product_attention(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    config: LlamaConfig,
    mode: str,
) -> jnp.ndarray:
    """
    query/key/value layout:
        [B, T, H, Dh]

    attention_mask layout:
        [1, 1, Tq, Tk]

    jax.nn.dot_product_attention supports GQA/MQA directly when
    Q heads are divisible by KV heads, so we do not repeat KV heads.
    """

    # Convert existing mask layout [1, 1, Tq, Tk]
    # to bias layout broadcastable against [B, H, Tq, Tk]
    # used internally by SDPA implementations.
    bias = attention_mask

    requested_impl = _attention_impl_from_config(
        config,
        mode,
    )

    if requested_impl == "cudnn":
        try:
            _log_attention_backend("cudnn dot_product_attention")

            return jax.nn.dot_product_attention(
                query_states,
                key_states,
                value_states,
                bias=bias,
                is_causal=False,
                implementation="cudnn",
            )

        except Exception as e:
            warnings.warn(
                "[attention] cuDNN attention failed "
                f"({type(e).__name__}: {e}); falling back to XLA.",
                RuntimeWarning,
            )

    _log_attention_backend("xla dot_product_attention")

    return jax.nn.dot_product_attention(
        query_states,
        key_states,
        value_states,
        bias=bias,
        is_causal=False,
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
    ) -> tuple[jnp.ndarray, Optional[KVCache]]:

        hidden_states = constrain_hidden_states(hidden_states)

        config = self.config

        B, T, _ = hidden_states.shape

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

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

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        # [B, T, H, Dh]
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

        rotary_emb = RotaryEmbedding(config)

        cos, sin = rotary_emb(
            query_states,
            positions,
        )

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        key_states = constrain_kv_cache(key_states)
        value_states = constrain_kv_cache(value_states)

        updated_cache = None

        if kv_cache is not None:
            updated_cache, key_states, value_states = update_kv_cache(
                kv_cache,
                key_states,
                value_states,
            )

            kv_length = updated_cache.cache_position

            key_states = key_states[:, :kv_length, :, :]
            value_states = value_states[:, :kv_length, :, :]

        query_states = query_states.astype(config.compute_dtype)
        key_states = key_states.astype(config.compute_dtype)
        value_states = value_states.astype(config.compute_dtype)

        attn_output = _dot_product_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            config=config,
            mode=mode,
        )

        # [B, T, H, Dh] -> [B, T, D]
        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        attn_output = constrain_hidden_states(attn_output)

        attn_output = o_proj(attn_output)

        attn_output = constrain_hidden_states(attn_output)

        return attn_output, updated_cache