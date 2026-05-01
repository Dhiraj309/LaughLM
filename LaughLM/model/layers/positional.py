"""
LaughLM/model/layers/positional.py

Positional encoding layers for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. RoPE tables computed in float64 — prevents precision loss at long
   sequences. With float32, position 8192 × freq pairs accumulate
   enough error to degrade attention patterns. Float64 eliminates this.
   Tables are cast to float32 after computation for runtime use.

2. NTK-aware Scaled RoPE — when config selects rope_scaled, uses
   NTK-aware interpolation (Reddit/kaiokendev) for context extension.
   Changes theta from 10k → scaled_theta, enabling extrapolation
   beyond training length without fine-tuning.

3. Configurable theta — build_rope_tables accepts theta from config.
   Standard: 10_000 (≤8K context). Llama 3: 500_000 (128K context).

4. Cleaner apply_rope — same algorithm, clearer documentation.

References:
  RoPE: Su et al. "RoFormer" (2021)
  NTK-aware: kaiokendev (Reddit, 2023) + Code Llama
  Llama 3: theta=500_000 for 128K context
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig


# ════════════════════════════════════════════════════════════════
# Learned Positional Embeddings (GPT-2 style)
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# Sinusoidal Positional Embeddings (original Transformer)
# ════════════════════════════════════════════════════════════════

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Deterministic sinusoidal embeddings (Vaswani et al., 2017).
    No learned parameters.
    """

    max_seq_len: int
    hidden_size: int

    def setup(self):
        position = jnp.arange(self.max_seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.hidden_size, 2)
            * -(jnp.log(10000.0) / self.hidden_size)
        )

        pe = jnp.zeros((self.max_seq_len, self.hidden_size))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        positions: [B, T]
        returns:   [B, T, hidden_size]
        """
        return self.pe[positions]


# ════════════════════════════════════════════════════════════════
# RoPE: Pre-compute sin/cos tables
# ════════════════════════════════════════════════════════════════

def build_rope_tables(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    scale_factor: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pre-compute RoPE sin/cos tables.

    Returns sin, cos each of shape [max_seq_len, head_dim // 2].

    These are computed ONCE in GPTModel.setup() and passed through
    TransformerBlock → attention, sliced to actual sequence length.

    Parameters
    ----------
    head_dim     : dimension per attention head
    max_seq_len  : maximum sequence length
    theta        : RoPE base frequency
                   10_000  → standard (≤8K context)
                   500_000 → Llama 3 style (128K context)
    scale_factor : NTK-aware scaling factor for context extension.
                   If provided, theta is scaled by:
                     theta_scaled = theta * scale_factor^(head_dim / (head_dim - 2))
                   This allows extrapolation beyond training length.
                   Typical value: desired_len / training_len (e.g. 4.0 for 4× extension).

    PRECISION NOTE: Tables are computed in float64 then cast to float32.
    At position 8192 with head_dim=128, float32 outer product accumulates
    ~1e-4 relative error in the angle values. Float64 keeps error < 1e-12.
    This matters for attention pattern quality at long sequences.

    Reference:
      Standard RoPE: Su et al. "RoFormer" (2021)
      NTK-aware:     kaiokendev (2023), used by Code Llama
    """
    import numpy as np  # Use numpy float64 for precision

    # ── NTK-aware theta scaling ──
    if scale_factor is not None and scale_factor > 1.0:
        theta = theta * (scale_factor ** (head_dim / (head_dim - 2)))

    # ── Frequency for each pair of dimensions ──
    # [head_dim // 2] in float64
    dim_idx = np.arange(0, head_dim, 2, dtype=np.float64)
    freqs = 1.0 / (theta ** (dim_idx / head_dim))

    # ── Position indices ──
    positions = np.arange(max_seq_len, dtype=np.float64)

    # ── Outer product: [max_seq_len, head_dim // 2] ──
    angles = np.outer(positions, freqs)

    # ── Compute sin/cos in float64, then cast to float32 for runtime ──
    sin = jnp.array(np.sin(angles), dtype=jnp.float32)
    cos = jnp.array(np.cos(angles), dtype=jnp.float32)

    return sin, cos


# ════════════════════════════════════════════════════════════════
# RoPE: Apply to Q or K tensor
# ════════════════════════════════════════════════════════════════

def apply_rope(
    x: jnp.ndarray,
    sin: jnp.ndarray,
    cos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply Rotary Position Embeddings to a Q or K tensor.

    Parameters
    ----------
    x   : [batch, seq_len, num_heads, head_dim]
    sin : [seq_len, head_dim // 2]  (sliced to current seq_len)
    cos : [seq_len, head_dim // 2]

    Returns
    -------
    x_rotated : [batch, seq_len, num_heads, head_dim]

    How RoPE works
    ──────────────
    Each pair of dimensions (x[..., 2i], x[..., 2i+1]) is treated as a
    2D vector and rotated by angle θ_i × position:

        out[..., 2i]   = x[..., 2i]   * cos - x[..., 2i+1] * sin
        out[..., 2i+1] = x[..., 2i+1] * cos + x[..., 2i]   * sin
    """
    # Split even/odd dimensions
    x_even = x[..., ::2]    # [B, T, H, D/2]
    x_odd  = x[..., 1::2]   # [B, T, H, D/2]

    # Reshape sin/cos for broadcasting: [1, T, 1, D/2]
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]

    # Apply rotation
    out_even = x_even * cos - x_odd  * sin
    out_odd  = x_odd  * cos + x_even * sin

    # Interleave back
    out = jnp.stack([out_even, out_odd], axis=-1)
    return out.reshape(x.shape)


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

def build_positional_encoding(
    config: LaughLMConfig,
) -> Optional[nn.Module]:
    """
    Build positional encoding module from config.

    Returns None for RoPE/ALiBi — those are applied inside attention,
    not as additive embeddings at the model input.
    """
    pos_type = config.architecture.positional
    max_seq  = config.model.max_seq_len
    hidden   = config.model.d_model

    if pos_type == "learned":
        return LearnedPositionalEmbedding(max_seq, hidden)

    if pos_type == "sinusoidal":
        return SinusoidalPositionalEmbedding(max_seq, hidden)

    if pos_type in ("rope", "rope_scaled"):
        # RoPE tables built in GPTModel.setup() via build_rope_tables().
        # Threaded through TransformerBlock → attention as (sin, cos).
        return None

    if pos_type == "alibi":
        return None

    raise ValueError(
        f"Unknown positional type: '{pos_type}'. "
        f"Valid options: learned, sinusoidal, rope, rope_scaled, alibi."
    )
