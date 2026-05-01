import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import dot_product_attention
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
    return jnp.transpose(x, (0, 2, 1, 3))  # (B, H, T, D)


def merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    b, h, t, d = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))  # (B, T, H, D)
    return x.reshape(b, t, h * d)


# ------------------------------------------------------------
# Attention Core (Flax kernel)
# ------------------------------------------------------------

def attention(q, k, v):
    return dot_product_attention(
        q,
        k,
        v,
        deterministic=True,
    )


# ------------------------------------------------------------
# RoPE (NO TRANSPOSE VERSION)
# ------------------------------------------------------------

def apply_rope_safe(q, k, sin, cos):
    # Assumes apply_rope supports (B, H, T, D)
    q = apply_rope(q, sin, cos)
    k = apply_rope(k, sin, cos)
    return q, k


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
        scale = head_dim ** -0.5

        # --------------------------------------------------------
        # QKV projection
        # --------------------------------------------------------
        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        # --------------------------------------------------------
        # RoPE FIRST (correct ordering)
        # --------------------------------------------------------
        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos)

        # --------------------------------------------------------
        # Scale AFTER RoPE
        # --------------------------------------------------------
        q = q * scale

        # --------------------------------------------------------
        # Attention
        # --------------------------------------------------------
        out = attention(q, k, v)

        # --------------------------------------------------------
        # Merge heads
        # --------------------------------------------------------
        out = merge_heads(out)

        # --------------------------------------------------------
        # Output projection
        # --------------------------------------------------------
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
        scale = head_dim ** -0.5

        proj_dim = self.d_model + 2 * head_dim

        qkv = nn.Dense(
            proj_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads)

        # single KV head
        k = k[:, :, None, :]
        v = v[:, :, None, :]

        k = jnp.transpose(k, (0, 2, 1, 3))  # (B, 1, T, D)
        v = jnp.transpose(v, (0, 2, 1, 3))

        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos)

        q = q * scale

        out = attention(q, k, v)
        out = merge_heads(out)

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
        scale = head_dim ** -0.5

        kv_dim = self.num_kv_heads * head_dim

        qkv = nn.Dense(
            self.d_model + 2 * kv_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_kv_heads)
        v = split_heads(v, self.num_kv_heads)

        if rope_tables is not None:
            sin, cos = rope_tables
            q, k = apply_rope_safe(q, k, sin, cos)

        q = q * scale

        # --------------------------------------------------------
        # Broadcast instead of repeat (CRITICAL FIX)
        # --------------------------------------------------------
        repeat = self.num_heads // self.num_kv_heads

        k = jnp.broadcast_to(
            k,
            (k.shape[0], self.num_heads, k.shape[2], k.shape[3]),
        )

        v = jnp.broadcast_to(
            v,
            (v.shape[0], self.num_heads, v.shape[2], v.shape[3]),
        )

        out = attention(q, k, v)
        out = merge_heads(out)

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
