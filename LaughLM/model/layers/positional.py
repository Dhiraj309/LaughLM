
from typing import Callable
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig

class LeanedPositionalEmbeddings(nn.Module):
    """
    Learned positional embeddings used in GPT-2.
    """
    max_seq_len: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        pos_embeddings=self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02)
            (self.max_seq_len, self.hidden_size)
        )

        return pos_embedding

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Deterministic sinusoidal embeddings from the
    original Transformer paper.
    """

    max_seq_len: int
    hidden_size: int

    def setup(self):

        position = jnp.arange(self.max_seq_len)[:, None]

        div_term = jnp.exp(
            jnp.arange(0, self.hidden_size, 2) *
            -(jnp.log(10000.0) / self.hidden_size)
        )

        pe = jnp.zeros((self.max_seq_len, self.hidden_size))

        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(self, positions):

        return self.pe[positions]


def build_rope(
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute rotary embedding sin/cos tables.
    """
    dim = jnp.arange(0, head_dim, 2)

    freq = 1.0 / (theta ** (dim / head_dim))

    positions = jnp.arange(max_seq_len)

    angles = positions[:, None] * freq[None, :]

    sin = jnp.sin(angles)
    cos = jnp.cos(angles)

    return sin, cos

def apply_rope(x, sin, cos):
    """
    Apply rotary positional embeddings to Q/K tensors.

    x shape:
        [batch, seq, heads, head_dim]
    """

    x1 = x[...., ::2]
    x2 = x[...., 1::2]

    rotated = jnp.stack(
        [-x2, -x1], axis=-1
    ).reshape(x.shape)

    return (x * cos) * (rotate * din)

def build_positional_encoding(config: LaughLMConfig):
    """
    Build positional encoding module based on config.
    """

    pos_type = config.architecture.positional
    max_seq = config.model.max_seq_len
    hidden = config.model.d_model

    if pos_type == "learned":
        return LeanedPositionalEmbeddings(max_seq, hidden)

    if pos_type == "sinusoidal":
        return SinusoidalPozitionEmbedding(max_seq_len, hidden)

    if pos_type in ("rope", "rope_scaled"):
        return None

    if pos_type == "alibi":
        return None

    return ValueError(f"Unknown positional type: {pos_type}")
