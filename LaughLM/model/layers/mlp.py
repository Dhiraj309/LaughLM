import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Activations
# ------------------------------------------------------------

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian Error Linear Unit (tanh approximation).
    Used in GPT-2, BERT, original Transformer FFN.
    """
    return 0.5 * x * (
        1.0 + jnp.tanh(
            jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)
        )
    )


def swish(x: jnp.ndarray) -> jnp.ndarray:
    """
    Swish / SiLU activation: x * sigmoid(x).
    Used as the gate function in SwiGLU.

    FIX: was 'nn.sigmoid' (flax.linen has no sigmoid function).
    Now correctly uses jax.nn.sigmoid.
    """
    # jax.nn.sigmoid is the correct namespace
    return x * jax.nn.sigmoid(x)


# ------------------------------------------------------------
# FFN: Standard GELU MLP
# ------------------------------------------------------------

class GELUMLP(nn.Module):
    """
    Standard two-layer MLP with GELU activation.
    Used in GPT-2 and BERT. Superseded by SwiGLU in modern models.

    Parameter count: 2 × d_model × ffn_dim = 8 × d_model² (with ffn_dim = 4 × d_model)
    """

    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.ffn_dim, use_bias=self.use_bias)(x)
        x = gelu(x)
        x = nn.Dense(self.d_model, use_bias=self.use_bias)(x)
        return x


# ------------------------------------------------------------
# FFN: GEGLU
# ------------------------------------------------------------

class GEGLU(nn.Module):
    """
    Gated GELU: element-wise product of two parallel projections.
    One branch is activated with GELU (gate), the other is linear (value).

    GLU variant from Noam Shazeer, 2020.
    Parameter count: 3 × d_model × ffn_dim (gate, value, down)
    Set ffn_dim = int(8/3 × d_model) to match 2-layer MLP parameter budget.
    """

    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate  = nn.Dense(self.ffn_dim, use_bias=self.use_bias)(x)
        value = nn.Dense(self.ffn_dim, use_bias=self.use_bias)(x)
        x = gelu(gate) * value
        x = nn.Dense(self.d_model, use_bias=self.use_bias)(x)
        return x


# ------------------------------------------------------------
# FFN: SwiGLU (recommended for all modern models)
# ------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    Gated FFN with Swish activation.

    Architecture (Shazeer 2020, used in PaLM, LLaMA, Mistral, DeepSeek):
        gate_proj:  d_model → ffn_dim  (then Swish)
        up_proj:    d_model → ffn_dim  (linear)
        down_proj:  ffn_dim → d_model

        output = down_proj(swish(gate_proj(x)) * up_proj(x))
    """

    d_model:  int
    ffn_dim:  int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        # Fused gate/value projection

        proj = nn.Dense(
            2 * self.ffn_dim,
            use_bias=self.use_bias
        )(x)

        gate, value  = jnp.split(proj, 2, axis=-1)   # gate branch value branch

        x = swish(gate) * value                                      # gated activation

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias
            )(x)       # project back
        return x


# ------------------------------------------------------------
# Utility: compute ffn_dim aligned to hardware
# ------------------------------------------------------------

def compute_ffn_dim(d_model: int, ffn_type: str, multiple_of: int = 64) -> int:
    """
    Compute FFN hidden dimension, hardware-aligned to `multiple_of`.

    For SwiGLU/GEGLU: use 8/3 × d_model to match 4× standard MLP param budget.
    For GELU MLP: use 4 × d_model (standard).

    TPU matrix multiplications are most efficient when all dimensions
    are multiples of 64 (or 128 for very large models).
    This rounding gives ~2-5% free throughput improvement.
    """

    if ffn_type in ("swiglu", "geglu"):
        # 8/3 × d_model for parameter parity with standard 2-matrix MLP
        raw_dim = int(8 / 3 * d_model)
    else:
        # Standard 4× for GELU MLP
        raw_dim = 4 * d_model

    # Round UP to nearest multiple_of
    aligned = ((raw_dim + multiple_of - 1) // multiple_of) * multiple_of

    return aligned


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_mlp(config: LaughLMConfig) -> nn.Module:

    ffn_type = config.architecture.ffn_type
    d_model  = config.model.d_model
    use_bias = config.architecture.bias

    # FIX: compute correctly aligned ffn_dim (was hardcoded 4 × d_model for all types)
    ffn_dim = compute_ffn_dim(d_model, ffn_type, multiple_of=64)

    if ffn_type == "gelu_mlp":
        return GELUMLP(d_model, ffn_dim, use_bias)

    if ffn_type == "geglu":
        return GEGLU(d_model, ffn_dim, use_bias)

    if ffn_type == "swiglu":
        return SwiGLU(d_model, ffn_dim, use_bias)

    if ffn_type == "moe":
        raise NotImplementedError(
            "MoE FFN is not yet implemented. Use 'swiglu'."
        )

    raise ValueError(f"Unknown FFN type: '{ffn_type}'")