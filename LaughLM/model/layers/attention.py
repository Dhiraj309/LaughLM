import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope


# ------------------------------------------------------------
# Utility: reshape Q/K/V
# ------------------------------------------------------------

def split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    """
    [B, T, D] → [B, T, num_heads, head_dim]
    """
    b, t, d = x.shape
    head_dim = d // num_heads
    return x.reshape(b, t, num_heads, head_dim)


def merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    """
    [B, T, num_heads, head_dim] → [B, T, D]
    """
    b, t, h, d = x.shape
    return x.reshape(b, t, h * d)


# ------------------------------------------------------------
# Causal mask builder
# ------------------------------------------------------------

def build_causal_mask(
    seq_len: int,
    dtype: jnp.dtype,
    doc_ids: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Build additive causal attention bias.

    Returns a [1, 1, T, T] (or [B, 1, T, T] when doc_ids provided) tensor:
        0.0  where attention is allowed
        -inf where attention is blocked

    Two blocking conditions:
      1. Future positions (j > i) — standard causal masking
      2. Cross-document positions — when doc_ids[i] != doc_ids[j]
         Required when sequence packing is used (multiple docs per sequence).
         Without this, the model learns to use context from previous documents
         which is information it won't have at inference time.

    Parameters
    ----------
    seq_len  : current sequence length T
    dtype    : dtype of attention scores (for -inf representation)
    doc_ids  : [B, T] integer document IDs. None = single document per sequence.
    """

    # -inf value safe for bfloat16 and float32
    neg_inf = jnp.finfo(dtype).min / 2

    # Triangular causal mask: True = attend, False = block
    # Shape [T, T] → broadcast to [1, 1, T, T]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    if doc_ids is not None:
        # doc_ids: [B, T]
        # same_doc[b, i, j] = True when tokens i and j are in the same document
        same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]   # [B, T, T]

        # Combine: attend only if BOTH causal and same document
        # Broadcast causal [T, T] with same_doc [B, T, T]
        mask = causal[None, :, :] & same_doc                     # [B, T, T]

        # Expand for heads: [B, 1, T, T]
        mask = mask[:, None, :, :]

    else:
        # No packing: simple causal mask, broadcast over batch and heads
        mask = causal[None, None, :, :]                           # [1, 1, T, T]

    # Convert bool mask to additive float bias
    # True (allow) → 0.0, False (block) → -inf
    attn_bias = jnp.where(mask, jnp.zeros_like(mask, dtype=dtype), neg_inf)

    return attn_bias


# ------------------------------------------------------------
# Multi-Head Attention (MHA)
# ------------------------------------------------------------

class MultiHeadAttention(nn.Module):

    d_model:   int
    num_heads: int
    use_bias:  bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model, use_bias=self.use_bias)(x)
        k = nn.Dense(self.d_model, use_bias=self.use_bias)(x)
        v = nn.Dense(self.d_model, use_bias=self.use_bias)(x)

        q = split_heads(q, self.num_heads)   # [B, T, H, Dh]
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        # Apply RoPE to Q and K
        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        # Attention scores: [B, H, T, T]
        attn_scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(head_dim)

        # Causal + cross-document mask
        attn_bias = build_causal_mask(
            seq_len=x.shape[1],
            dtype=attn_scores.dtype,
            doc_ids=doc_ids,
        )
        attn_scores = attn_scores + attn_bias

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("bhts,bshd->bthd", attn_probs, v)
        out = merge_heads(out)
        out = nn.Dense(self.d_model, use_bias=self.use_bias)(out)

        return out


# ------------------------------------------------------------
# Multi-Query Attention (MQA)
# ------------------------------------------------------------

class MultiQueryAttention(nn.Module):

    d_model:   int
    num_heads: int
    use_bias:  bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model, use_bias=self.use_bias)(x)     # [B, T, D]
        k = nn.Dense(head_dim,     use_bias=self.use_bias)(x)     # [B, T, Dh] — single head
        v = nn.Dense(head_dim,     use_bias=self.use_bias)(x)

        q = split_heads(q, self.num_heads)                         # [B, T, H, Dh]

        # Single KV head: expand dims for broadcasting
        k = k[:, :, None, :]                                       # [B, T, 1, Dh]
        v = v[:, :, None, :]

        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        attn_scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(head_dim)

        attn_bias = build_causal_mask(x.shape[1], attn_scores.dtype, doc_ids)
        attn_scores = attn_scores + attn_bias

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("bhts,bshd->bthd", attn_probs, v)
        out = merge_heads(out)
        out = nn.Dense(self.d_model, use_bias=self.use_bias)(out)

        return out


# ------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ------------------------------------------------------------

class GroupedQueryAttention(nn.Module):

    d_model:      int
    num_heads:    int
    num_kv_heads: int
    use_bias:     bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        GQA: num_heads query heads share num_kv_heads KV heads.
        repeat = num_heads // num_kv_heads queries per KV group.

        Example (Llama 3 70B style): 64 Q heads, 8 KV heads → 8 groups of 8.
        Example (this config):       12 Q heads, 4 KV heads → 4 groups of 3.
        """

        head_dim = self.d_model // self.num_heads
        kv_dim   = self.num_kv_heads * head_dim

        # q = nn.Dense(self.d_model, use_bias=self.use_bias)(x)   # [B, T, D]
        # k = nn.Dense(kv_dim,       use_bias=self.use_bias)(x)   # [B, T, kv_dim]
        # v = nn.Dense(kv_dim,       use_bias=self.use_bias)(x)

        # q = split_heads(q, self.num_heads)       # [B, T, H,   Dh]
        # k = split_heads(k, self.num_kv_heads)    # [B, T, Hkv, Dh]
        # v = split_heads(v, self.num_kv_heads)

        qkv = nn.Dense(3 * self.d_model, use_bias=self.use_bias)(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_kv_heads)
        v = split_heads(v, self.num_kv_heads)

        # Apply RoPE to Q and K BEFORE expanding KV groups
        # (more efficient: rotate the compact KV, then expand)
        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        # Expand KV heads to match Q heads by repeating each KV group
        repeat = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)        # [B, T, H, Dh]
        v = jnp.repeat(v, repeat, axis=2)

        # Scaled dot-product attention
        attn_scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(head_dim)

        attn_bias = build_causal_mask(x.shape[1], attn_scores.dtype, doc_ids)
        attn_scores = attn_scores + attn_bias

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("bhts,bshd->bthd", attn_probs, v)
        out = merge_heads(out)
        out = nn.Dense(self.d_model, use_bias=self.use_bias)(out)

        return out


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_attention(config: LaughLMConfig) -> nn.Module:

    d_model   = config.model.d_model
    num_heads = config.model.num_heads
    use_bias  = config.architecture.bias
    variant   = config.architecture.attention_variant

    if variant == "mha":
        return MultiHeadAttention(d_model, num_heads, use_bias)

    if variant == "mqa":
        return MultiQueryAttention(d_model, num_heads, use_bias)

    if variant == "gqa":
        # num_kv_heads is validated by validation.py to be set and valid
        num_kv_heads = config.model.num_kv_heads
        return GroupedQueryAttention(d_model, num_heads, num_kv_heads, use_bias)

    if variant == "mla":
        raise NotImplementedError(
            "MLA (Multi-head Latent Attention) is not yet implemented. "
            "Use 'gqa' for efficient attention."
        )

    raise ValueError(f"Unknown attention variant: '{variant}'")