import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def split_heads(x, num_heads):
    b, t, d = x.shape
    return x.reshape(b, t, num_heads, d // num_heads)


def merge_heads(x):
    b, t, h, d = x.shape
    return x.reshape(b, t, h * d)


# ------------------------------------------------------------
# Flash Attention (FINAL: padded + stable)
# ------------------------------------------------------------

def flash_attention(q, k, v, mask=None, block_size=128):
    """
    q, k, v: [B, T, H, Dh]
    mask: [B, 1, T, T] or [1, 1, T, T]
    """

    B, T, H, Dh = q.shape
    attn_scale = 1.0 / jnp.sqrt(Dh)

    # ------------------------------------------------------------
    # PAD to multiple of block_size
    # ------------------------------------------------------------
    pad_len = (block_size - (T % block_size)) % block_size

    if pad_len > 0:
        pad_q = jnp.zeros((B, pad_len, H, Dh), dtype=q.dtype)
        pad_k = jnp.zeros((B, pad_len, H, Dh), dtype=k.dtype)
        pad_v = jnp.zeros((B, pad_len, H, Dh), dtype=v.dtype)

        q = jnp.concatenate([q, pad_q], axis=1)
        k = jnp.concatenate([k, pad_k], axis=1)
        v = jnp.concatenate([v, pad_v], axis=1)

        if mask is not None:
            neg_inf = jnp.finfo(mask.dtype).min / 2

            # pad columns
            mask_pad_col = jnp.full(
                (mask.shape[0], mask.shape[1], T, pad_len),
                neg_inf,
                dtype=mask.dtype,
            )
            mask = jnp.concatenate([mask, mask_pad_col], axis=-1)

            # pad rows
            mask_pad_row = jnp.full(
                (mask.shape[0], mask.shape[1], pad_len, T + pad_len),
                neg_inf,
                dtype=mask.dtype,
            )
            mask = jnp.concatenate([mask, mask_pad_row], axis=-2)

    T_padded = q.shape[1]
    num_blocks = T_padded // block_size

    # ------------------------------------------------------------
    # Flash attention core
    # ------------------------------------------------------------

    def body_q(bi, out):
        i = bi * block_size

        q_block = jax.lax.dynamic_slice(q, (0, i, 0, 0), (B, block_size, H, Dh))

        m = jnp.full((B, H, block_size), -1e30)
        l = jnp.zeros((B, H, block_size))
        acc = jnp.zeros((B, block_size, H, Dh))

        def body_k(bj, state):
            m, l, acc = state
            j = bj * block_size

            k_block = jax.lax.dynamic_slice(k, (0, j, 0, 0), (B, block_size, H, Dh))
            v_block = jax.lax.dynamic_slice(v, (0, j, 0, 0), (B, block_size, H, Dh))

            scores = jnp.einsum("bqhd,bkhd->bhqk", q_block, k_block) * attn_scale

            if mask is not None:
                mask_block = jax.lax.dynamic_slice(
                    mask,
                    (0, 0, i, j),
                    (mask.shape[0], mask.shape[1], block_size, block_size),
                )
                scores = scores + mask_block

            # stable softmax
            scores_max = jnp.max(scores, axis=-1)
            m_new = jnp.maximum(m, scores_max)

            exp_scores = jnp.exp(scores - m_new[..., None])
            l_new = jnp.sum(exp_scores, axis=-1)

            norm = l / l_new
            norm = jnp.transpose(norm, (0, 2, 1))  # [B, Qb, H]

            l_new_t = jnp.transpose(l_new, (0, 2, 1))

            acc = acc * norm[..., None] + jnp.einsum(
                "bhqk,bkhd->bqhd", exp_scores, v_block
            ) / l_new_t[..., None]

            return (m_new, l_new, acc)

        m, l, acc = jax.lax.fori_loop(0, num_blocks, body_k, (m, l, acc))

        return jax.lax.dynamic_update_slice(out, acc, (0, i, 0, 0))

    out = jnp.zeros_like(q)
    out = jax.lax.fori_loop(0, num_blocks, body_q, out)

    # ------------------------------------------------------------
    # UNPAD
    # ------------------------------------------------------------
    return out[:, :T, :, :]


# ------------------------------------------------------------
# Mask
# ------------------------------------------------------------

def build_causal_mask(seq_len, dtype, doc_ids=None):
    neg_inf = jnp.finfo(dtype).min / 2
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    if doc_ids is not None:
        same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
        mask = causal[None, :, :] & same_doc
        mask = mask[:, None, :, :]
    else:
        mask = causal[None, None, :, :]

    return jnp.where(mask, 0.0, neg_inf)


# ------------------------------------------------------------
# MHA
# ------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        qkv = nn.Dense(3 * self.d_model, use_bias=self.use_bias)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        mask = build_causal_mask(x.shape[1], q.dtype, doc_ids)

        out = jax.nn.dot_product_attention(
            q, k, v,
            bias=mask,
            is_causal=False,
        )
        out = merge_heads(out)

        return nn.Dense(self.d_model, use_bias=self.use_bias)(out)


# ------------------------------------------------------------
# MQA
# ------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    d_model: int
    num_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        head_dim = self.d_model // self.num_heads

        proj_dim = self.d_model + 2 * head_dim
        qkv = nn.Dense(proj_dim, use_bias=self.use_bias)(x)

        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads)
        k = k[:, :, None, :]
        v = v[:, :, None, :]

        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        mask = build_causal_mask(x.shape[1], q.dtype, doc_ids)

        out = jax.nn.dot_product_attention(            q, k, v,
            bias=mask,
            is_causal=False,
        )
        out = merge_heads(out)

        return nn.Dense(self.d_model, use_bias=self.use_bias)(out)


# ------------------------------------------------------------
# GQA
# ------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    d_model: int
    num_heads: int
    num_kv_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None):

        head_dim = self.d_model // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        qkv = nn.Dense(self.d_model + 2 * kv_dim, use_bias=self.use_bias)(x)
        q, kv = jnp.split(qkv, [self.d_model], axis=-1)
        k, v = jnp.split(kv, 2, axis=-1)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_kv_heads)
        v = split_heads(v, self.num_kv_heads)

        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        repeat = self.num_heads // self.num_kv_heads

        k = jnp.broadcast_to(
            k[:, :, :, None, :],
            (k.shape[0], k.shape[1], self.num_kv_heads, repeat, head_dim),
        ).reshape(k.shape[0], k.shape[1], self.num_heads, head_dim)

        v = jnp.broadcast_to(
            v[:, :, :, None, :],
            (v.shape[0], v.shape[1], self.num_kv_heads, repeat, head_dim),
        ).reshape(v.shape[0], v.shape[1], self.num_heads, head_dim)

        mask = build_causal_mask(x.shape[1], q.dtype, doc_ids)

        out = jax.nn.dot_product_attention(            q, k, v,
            bias=mask,
            is_causal=False,
        )
        out = merge_heads(out)

        return nn.Dense(self.d_model, use_bias=self.use_bias)(out)


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_attention(config: LaughLMConfig):

    variant = config.architecture.attention_variant

    if variant == "mha":
        return MultiHeadAttention(
            config.model.d_model,
            config.model.num_heads,
            config.architecture.bias,
        )

    if variant == "mqa":
        return MultiQueryAttention(
            config.model.d_model,
            config.model.num_heads,
            config.architecture.bias,
        )

    if variant == "gqa":
        return GroupedQueryAttention(
            config.model.d_model,
            config.model.num_heads,
            config.model.num_kv_heads,
            config.architecture.bias,
        )

    raise ValueError(f"Unknown attention variant: {variant}")
