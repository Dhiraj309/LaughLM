import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope
from LaughLM.utils.dtype import get_dtype


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    b, t, d = x.shape
    head_dim = d // num_heads
    x = x.reshape(b, t, num_heads, head_dim)
    return jnp.transpose(x, (0, 2, 1, 3))  # (b, h, t, d)


def merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    b, h, t, d = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))  # (b, t, h, d)
    return x.reshape(b, t, h * d)


# ------------------------------------------------------------
# Attention Core
# ------------------------------------------------------------

def attention(q, k, v):
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
    )


# ------------------------------------------------------------
# RoPE Helper (layout-safe)
# ------------------------------------------------------------

def apply_rope_safe(q, k, sin, cos, compute_dtype):
    # (B, H, T, D) → (B, T, H, D)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))

    q = apply_rope(q, sin, cos)
    k = apply_rope(k, sin, cos)

    # back to (B, H, T, D)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))

    return q.astype(compute_dtype), k.astype(compute_dtype)


# ------------------------------------------------------------
# Multi-Head Attention (MHA)
# ------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    config: LaughLMConfig
    d_model: int
    num_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype = get_dtype(self.config.parallelism.param_dtype)

        head_dim = self.d_model // self.num_heads

        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = split_heads(q, self.num_heads).astype(compute_dtype)
        k = split_heads(k, self.num_heads).astype(compute_dtype)
        v = split_heads(v, self.num_heads).astype(compute_dtype)

        q = q * (head_dim ** -0.5)

        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos, compute_dtype)

        out = attention(q, k, v)
        out = merge_heads(out).astype(compute_dtype)

        out = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(out)

        return out


# ------------------------------------------------------------
# Multi-Query Attention (MQA)
# ------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    config: LaughLMConfig
    d_model: int
    num_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype = get_dtype(self.config.parallelism.param_dtype)

        head_dim = self.d_model // self.num_heads
        proj_dim = self.d_model + 2 * head_dim

        qkv = nn.Dense(
            proj_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads).astype(compute_dtype)

        k = k[:, :, None, :].astype(compute_dtype)
        v = v[:, :, None, :].astype(compute_dtype)

        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        q = q * (head_dim ** -0.5)

        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos, compute_dtype)

        out = attention(q, k, v)
        out = merge_heads(out).astype(compute_dtype)

        out = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(out)

        return out


# ------------------------------------------------------------
# Grouped Query Attention (GQA)
# ------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    config: LaughLMConfig
    d_model: int
    num_heads: int
    num_kv_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype = get_dtype(self.config.parallelism.param_dtype)

        head_dim = self.d_model // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        qkv = nn.Dense(
            self.d_model + 2 * kv_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads).astype(compute_dtype)
        k = split_heads(k, self.num_kv_heads).astype(compute_dtype)
        v = split_heads(v, self.num_kv_heads).astype(compute_dtype)

        q = q * (head_dim ** -0.5)

        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos, compute_dtype)

        repeat = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, repeat, axis=1)
        v = jnp.repeat(v, repeat, axis=1)

        out = attention(q, k, v)
        out = merge_heads(out).astype(compute_dtype)

        out = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(out)

        return out


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_attention(config: LaughLMConfig):

    variant = config.architecture.attention_variant

    if variant == "mha":
        return MultiHeadAttention(
            config=config,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            use_bias=config.architecture.bias,
        )

    if variant == "mqa":
        return MultiQueryAttention(
            config=config,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            use_bias=config.architecture.bias,
        )

    if variant == "gqa":
        return GroupedQueryAttention(
            config=config,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            num_kv_heads=config.model.num_kv_heads,
            use_bias=config.architecture.bias,
        )

    raise ValueError(f"Unknown attention variant: {variant}")
