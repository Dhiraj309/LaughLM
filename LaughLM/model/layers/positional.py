import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Learned Positional Embeddings (GPT-2 style)
# ------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.
    Hard cap at max_seq_len — cannot extrapolate beyond training length.
    """

    max_seq_len: int
    hidden_size: int

    @nn.compact
    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        positions: [B, T] integer positions (0..T-1)
        returns:   [B, T, hidden_size]
        """
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.hidden_size),
        )

        return pos_embedding[positions]


# ------------------------------------------------------------
# Sinusoidal Positional Embeddings (original Transformer)
# ------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Deterministic sinusoidal embeddings.
    No learned parameters. Cannot be updated by gradient descent.
    """

    max_seq_len: int
    hidden_size: int

    def setup(self):
        position = jnp.arange(self.max_seq_len)[:, None]         # [T, 1]

        div_term = jnp.exp(
            jnp.arange(0, self.hidden_size, 2) *
            -(jnp.log(10000.0) / self.hidden_size)
        )                                                          # [D/2]

        pe = jnp.zeros((self.max_seq_len, self.hidden_size))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe                                               # [T, D]

    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        positions: [B, T]
        returns:   [B, T, hidden_size]
        """
        # FIX: was SinusoidalPozitionEmbedding (typo) — class name corrected
        return self.pe[positions]


# ------------------------------------------------------------
# RoPE: Pre-compute sin/cos tables
# ------------------------------------------------------------

def build_rope_tables(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pre-compute RoPE sin/cos tables.

    Returns sin, cos each of shape [max_seq_len, head_dim // 2].

    These are computed ONCE in GPTModel.setup() and passed through
    TransformerBlock → attention at each forward call, sliced to
    the actual sequence length.

    theta=10_000 is standard for seq_len ≤ 8192.
    For longer contexts use theta=500_000 (Llama 3 style).
    """
    # Frequency for each pair of dimensions: [head_dim // 2]
    dim_idx = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    freqs   = 1.0 / (theta ** (dim_idx / head_dim))

    # Position indices: [max_seq_len]
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product: [max_seq_len, head_dim // 2]
    angles = jnp.outer(positions, freqs)

    sin = jnp.sin(angles)   # [max_seq_len, head_dim // 2]
    cos = jnp.cos(angles)   # [max_seq_len, head_dim // 2]

    return sin, cos


# ------------------------------------------------------------
# RoPE: Apply to Q or K tensor
# ------------------------------------------------------------

def apply_rope(
    x: jnp.ndarray,
    sin: jnp.ndarray,
    cos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply Rotary Position Embeddings to a Q or K tensor.

    Arguments
    ---------
    x   : [batch, seq_len, num_heads, head_dim]
    sin : [seq_len, head_dim // 2]   (sliced to current seq_len)
    cos : [seq_len, head_dim // 2]

    Returns
    -------
    x_rotated : [batch, seq_len, num_heads, head_dim]

    How RoPE works
    --------------
    RoPE treats each pair of dimensions (x[..., 2i], x[..., 2i+1]) as a
    2D vector and rotates it by angle θ_i × position.

    The rotation matrix for angle θ is:
        [cos θ,  -sin θ]
        [sin θ,   cos θ]

    Applied element-wise:
        out[..., 2i]   = x[..., 2i]   * cos[pos, i] - x[..., 2i+1] * sin[pos, i]
        out[..., 2i+1] = x[..., 2i+1] * cos[pos, i] + x[..., 2i]   * sin[pos, i]
    """

    # Split even/odd dimensions
    x_even = x[..., ::2]    # [B, T, H, D/2] — indices 0, 2, 4, ...
    x_odd  = x[..., 1::2]   # [B, T, H, D/2] — indices 1, 3, 5, ...

    # Reshape sin/cos for broadcasting: [1, T, 1, D/2]
    sin = sin[None, :, None, :]   # broadcast over batch and heads
    cos = cos[None, :, None, :]

    # Apply rotation
    out_even = x_even * cos - x_odd  * sin
    out_odd  = x_odd  * cos + x_even * sin

    # Interleave back: stack along last axis then reshape
    # jnp.stack([out_even, out_odd], axis=-1) → [B, T, H, D/2, 2]
    # .reshape(x.shape)                       → [B, T, H, D]
    out = jnp.stack([out_even, out_odd], axis=-1)
    return out.reshape(x.shape)


# ------------------------------------------------------------
# Factory: build positional encoding module (or None for RoPE)
# ------------------------------------------------------------

def build_positional_encoding(
    config: LaughLMConfig,
) -> Optional[nn.Module]:
    """
    Build positional encoding module from config.

    Returns None for RoPE/ALiBi — those are applied inside attention,
    not as additive embeddings at the model input.
    The caller (GPTModel) skips the additive step when None is returned.
    """

    pos_type = config.architecture.positional
    max_seq  = config.model.max_seq_len
    hidden   = config.model.d_model

    if pos_type == "learned":
        return LearnedPositionalEmbedding(max_seq, hidden)

    if pos_type == "sinusoidal":
        # FIX: was SinusoidalPozitionEmbedding + wrong variable name max_seq_len
        return SinusoidalPositionalEmbedding(max_seq, hidden)

    if pos_type in ("rope", "rope_scaled"):
        # RoPE tables are built in GPTModel.setup() via build_rope_tables().
        # They are threaded through TransformerBlock → attention as (sin, cos).
        # No additive embedding at the input — return None.
        return None

    if pos_type == "alibi":
        # ALiBi is also applied inside attention, not at input.
        return None

    # FIX: was 'return ValueError(...)' — must be 'raise'
    raise ValueError(
        f"Unknown positional type: '{pos_type}'. "
        f"Valid options: learned, sinusoidal, rope, rope_scaled, alibi."
    )