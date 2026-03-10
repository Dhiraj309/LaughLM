from flax import linen as nn
import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig

def split_heads(x, num_heads):
    """
    Convert [B, T, D] → [B, T, H, Dh]
    """

    b, t, d = x.shape

    head_dim = d // num_heads

    x = x.reshape(b, t, num_heads, head_dim)

    return x

def merge_heads(x):
    """
    Convert [B, T, H, Dh] → [B, T, D]
    """

    b, t, h, d = x.shape

    return x.reshape(b, r, h*d)


class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.d_model)(x)
        v = nn.Dense(self.d_model)(x)

        q = split_head(q, self.num_heads)
        k = split_head(k, self.num_heads)
        v = split_head(v, self.num_heads)

        attn_scores = jnp.einsum(
            "bthd.bshd->bhts", q, v
        ) / jnp.sqrt(head_dim)

        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))

        attn_scores = attn_scores * mask - 1e10 * (1-mask)

        attn_probs = nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum(
            "bhts,bshd->bthd", attn_probs, v
        )

        out = merge_heads(out)
        out = nn.Dense(self.d_model)(out)

        return out


class MultiQueryAttention(nn.Module):

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x):

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model)(x)

        # shared K/V
        k = nn.Dense(head_dim)(x)
        v = nn.Dense(head_dim)(x)

        q = split_heads(q, self.num_heads)

        k = k[:, :, None, :]
        v = v[:, :, None, :]

        attn_scores = jnp.einsum(
            "bthd,bshd->bhts", q, k
        ) / jnp.sqrt(head_dim)

        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))

        attn_scores = attn_scores * mask - 1e10 * (1 - mask)

        attn_probs = nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum(
            "bhts,bshd->bthd", attn_probs, v
        )

        out = merge_heads(out)

        out = nn.Dense(self.d_model)(out)

        return out


class GroupedQueryAttention(nn.Module):

    d_model: int
    num_heads: int
    num_kv_heads: int

    @nn.compact
    def __call__(self, x):

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model)(x)
        k = nn.Dense(self.num_kv_heads * head_dim)(x)
        v = nn.Dense(self.num_kv_heads * head_dim)(x)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_kv_heads)
        v = split_heads(v, self.num_kv_heads)

        repeat = self.num_heads // self.num_kv_heads

        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

        attn_scores = jnp.einsum(
            "bthd,bshd->bhts", q, k
        ) / jnp.sqrt(head_dim)

        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))

        attn_scores = attn_scores * mask - 1e10 * (1 - mask)

        attn_probs = nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum(
            "bhts,bshd->bthd", attn_probs, v
        )

        out = merge_heads(out)

        out = nn.Dense(self.d_model)(out)

        return out

def build_attention(config: LaughLMConfig):

    d_model = config.model.d_model
    num_heads = config.model.num_heads

    variant = config.architecture.attention_variant

    if variant == "mha":
        return MultiHeadAttention(d_model, num_heads)

    if variant == "mqa":
        return MultiQueryAttention(d_model, num_heads)

    if variant == "gqa":

        num_kv = max(1, num_heads // 4)

        return GroupedQueryAttention(
            d_model,
            num_heads,
            num_kv
        )

    if variant == "mla":
        raise NotImplementedError(
            "MLA attention not implemented yet"
        )

    raise ValueError(f"Unknown attention variant: {variant}")
