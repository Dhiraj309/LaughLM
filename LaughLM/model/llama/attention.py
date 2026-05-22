"""
LaughLM/model/llama/attention.py

Canonical LLaMA attention for PMAP production + TPU Splash experiment.

Backend policy:
- standard / xla / flash / cudnn / memory_efficient -> XLA SDPA
- splash -> TPU SplashAttention under PMAP/local execution, fallback XLA SDPA
- decode -> XLA SDPA
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


def _is_tpu_backend() -> bool:
    return jax.default_backend() == "tpu"


def _find_splash_block_size(seq_len: int) -> tuple[int, int]:
    for block in (512, 256, 128):
        if seq_len % block == 0:
            return block, 0

    block = 512
    pad = ((seq_len + block - 1) // block) * block - seq_len
    return block, pad


def _pad_for_splash(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    q/k/v layout:
        [B, T, H, Dh]
    """

    B, T, QH, Dh = q.shape
    KVH = k.shape[2]

    _, pad = _find_splash_block_size(T)

    if pad == 0:
        return q, k, v, 0

    q = jnp.concatenate(
        [
            q,
            jnp.zeros(
                (B, pad, QH, Dh),
                dtype=q.dtype,
            ),
        ],
        axis=1,
    )

    k = jnp.concatenate(
        [
            k,
            jnp.zeros(
                (B, pad, KVH, Dh),
                dtype=k.dtype,
            ),
        ],
        axis=1,
    )

    v = jnp.concatenate(
        [
            v,
            jnp.zeros(
                (B, pad, KVH, Dh),
                dtype=v.dtype,
            ),
        ],
        axis=1,
    )

    return q, k, v, pad


def _splash_attention(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
) -> jnp.ndarray:
    """
    TPU SplashAttention.

    Input/output layout:
        [B, T, H, Dh]

    Splash kernel layout:
        per-example [H, T, Dh]
    """

    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    q, k, v, pad_amount = _pad_for_splash(
        query_states,
        key_states,
        value_states,
    )

    B, T, QH, Dh = q.shape
    KVH = k.shape[2]

    if QH != KVH:
        raise NotImplementedError(
            "Splash path currently requires num_attention_heads == "
            "num_key_value_heads. Use XLA SDPA for GQA/MQA."
        )

    block, _ = _find_splash_block_size(T)

    _log_attention_backend(
        f"splash attention block={block} seq={T}"
    )

    # [B, T, H, Dh] -> [B, H, T, Dh]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    causal_mask = splash_attention_mask.CausalMask(
        shape=(T, T)
    )

    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(causal_mask,) * QH
    )

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        block_q_dq=block,
        block_kv_dq=block,
    )

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
    )

    def per_example(q_b, k_b, v_b):
        return splash_kernel(q_b, k_b, v_b, None)

    out = jax.vmap(per_example, in_axes=(0, 0, 0))(q, k, v)

    # [B, H, T, Dh] -> [B, T, H, Dh]
    out = jnp.transpose(out, (0, 2, 1, 3))

    if pad_amount > 0:
        out = out[:, :-pad_amount, :, :]

    return out


def _attention_impl_from_config(
    config: LlamaConfig,
    mode: str,
    q_len: int,
    kv_len: int,
) -> str:
    impl = getattr(config, "attention_impl", "standard")

    if mode == "decode":
        return "xla"

    if impl == "splash":
        if _is_tpu_backend() and q_len == kv_len and q_len > 4:
            return "splash"
        return "xla"

    return "xla"


def _xla_sdpa(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
) -> jnp.ndarray:
    _log_attention_backend("xla dot_product_attention")

    return jax.nn.dot_product_attention(
        query_states,
        key_states,
        value_states,
        bias=attention_mask,
        is_causal=False,
    )


def _attention(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    config: LlamaConfig,
    mode: str,
) -> jnp.ndarray:
    backend = _attention_impl_from_config(
        config=config,
        mode=mode,
        q_len=query_states.shape[1],
        kv_len=key_states.shape[1],
    )

    if backend == "splash":
        try:
            return _splash_attention(
                query_states,
                key_states,
                value_states,
            )
        except Exception as e:
            warnings.warn(
                "[attention] Splash failed "
                f"({type(e).__name__}: {e}); falling back to XLA SDPA.",
                RuntimeWarning,
            )

    return _xla_sdpa(
        query_states,
        key_states,
        value_states,
        attention_mask,
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

        attn_output = _attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            config=config,
            mode=mode,
        )

        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        attn_output = constrain_hidden_states(attn_output)

        attn_output = o_proj(attn_output)

        attn_output = constrain_hidden_states(attn_output)

        return attn_output, updated_cache
