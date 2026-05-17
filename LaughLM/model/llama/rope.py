"""
LaughLM/model/llama/rope.py

Canonical Llama Rotary Position Embeddings (RoPE).

Design goals:
- HF-compatible rotary semantics
- deterministic cache behavior
- explicit position handling
- minimal architecture surface
- future-compatible with:
    - KV cache decode
    - packed prefill
    - GQA
    - sharding

Tensor conventions
------------------
q, k:
    [batch, seq, heads, head_dim]

positions:
    [batch, seq]

cos, sin:
    [batch, seq, head_dim]

RoPE is applied BEFORE attention.
"""

from flax import linen as nn
import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate half the hidden dimensions.

    (x1, x2) -> (-x2, x1)

    Input:
        [..., head_dim]

    Output:
        [..., head_dim]
    """

    x1, x2 = jnp.split(x, 2, axis=-1)

    return jnp.concatenate(
        (-x2, x1),
        axis=-1,
    )


class RotaryEmbedding(nn.Module):
    """
    Canonical Llama rotary embedding.

    Produces cos/sin tensors from explicit positions.
    """

    config: LlamaConfig

    def setup(self):

        head_dim = self.config.head_dim

        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                jnp.arange(
                    0,
                    head_dim,
                    2,
                    dtype=jnp.float32,
                )
                / head_dim
            )
        )

        self.inv_freq = inv_freq

    def __call__(
        self,
        x: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate rotary cos/sin tensors.

        Parameters
        ----------
        x:
            Reference tensor for dtype/device.

            Shape:
                [B, T, H, Dh]

        positions:
            Explicit token positions.

            Shape:
                [B, T]

        Returns
        -------
        cos:
            [B, T, Dh]

        sin:
            [B, T, Dh]
        """

        positions = positions.astype(jnp.float32)

        freqs = jnp.einsum(
            "bt,d->btd",
            positions,
            self.inv_freq,
        )

        emb = jnp.concatenate(
            (freqs, freqs),
            axis=-1,
        )

        cos = jnp.cos(emb).astype(x.dtype)

        sin = jnp.sin(emb).astype(x.dtype)

        return cos, sin


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply rotary embeddings to q/k tensors.

    Parameters
    ----------
    q:
        [B, T, QH, Dh]

    k:
        [B, T, KVH, Dh]

    cos:
        [B, T, Dh]

    sin:
        [B, T, Dh]

    Returns
    -------
    q_embed:
        [B, T, QH, Dh]

    k_embed:
        [B, T, KVH, Dh]
    """

    cos = cos[:, :, None, :]

    sin = sin[:, :, None, :]

    q_embed = (
        (q * cos)
        + (rotate_half(q) * sin)
    )

    k_embed = (
        (k * cos)
        + (rotate_half(k) * sin)
    )

    return q_embed, k_embed