"""
LaughLM/model/layers/mlp.py

Feed-forward network layers for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Dtype from SPMD config — reads from config.spmd.dtype via
   resolve_compute_dtype / resolve_param_dtype instead of legacy fields.

2. Clamp safety — retained for bf16 overflow prevention in gated activations.

3. Precision annotation — jax.lax.Precision.DEFAULT instead of HIGH.
   HIGH forces float32 accumulation on TPU which halves MXU throughput.
   For bf16 training, DEFAULT gives hardware-native precision (bf16 accum
   on TPU, tf32 accum on A100) which is the standard for all frontier models.

4. FFN dim alignment — compute_ffn_dim aligns to multiple_of (default 64)
   for TPU tile efficiency. For SwiGLU/GEGLU uses 8/3 × d_model ratio.

References:
  SwiGLU: Shazeer "GLU Variants Improve Transformer" (2020)
  PaLM: Chowdhery et al. (2022) — uses SwiGLU with 8/3 ratio
  LLaMA: Touvron et al. (2023) — SwiGLU, no bias, 8/3 ratio
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.utils.dtype import resolve_compute_dtype, resolve_param_dtype


# ────────────────────────────────────────────────────────────────
# Stability: activation clamp (prevents bf16 overflow)
# ────────────────────────────────────────────────────────────────

def clamp(x: jnp.ndarray, limit: float = 30.0) -> jnp.ndarray:
    """Clamp activations to prevent bf16 overflow in gated units."""
    return jnp.clip(x, -limit, limit)


# ────────────────────────────────────────────────────────────────
# Activations
# ────────────────────────────────────────────────────────────────

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=True)


def swish(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.silu(x)


# ────────────────────────────────────────────────────────────────
# FFN: Standard GELU MLP
# ────────────────────────────────────────────────────────────────

class GELUMLP(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        x = nn.Dense(
            self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        x = gelu(clamp(x))

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        return x


# ────────────────────────────────────────────────────────────────
# FFN: GEGLU
# ────────────────────────────────────────────────────────────────

class GEGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        proj = nn.Dense(
            2 * self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        gate, value = jnp.split(proj, 2, axis=-1)
        gate = clamp(gate)
        x = gelu(gate) * value

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        return x


# ────────────────────────────────────────────────────────────────
# FFN: SwiGLU
# ────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        proj = nn.Dense(
            2 * self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        gate, value = jnp.split(proj, 2, axis=-1)
        gate = clamp(gate)
        x = swish(gate) * value

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )(x)

        return x


# ────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────

def compute_ffn_dim(d_model: int, ffn_type: str, multiple_of: int = 64) -> int:
    """
    Compute FFN intermediate dimension.

    SwiGLU/GEGLU: 8/3 × d_model (PaLM/LLaMA convention).
    GELU MLP:     4 × d_model (standard).

    Result is rounded up to multiple_of for TPU tile alignment.
    """
    if ffn_type in ("swiglu", "geglu"):
        raw_dim = int(8 / 3 * d_model)
    else:
        raw_dim = 4 * d_model

    aligned = ((raw_dim + multiple_of - 1) // multiple_of) * multiple_of
    return aligned


# ────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────

def build_mlp(config: LaughLMConfig) -> nn.Module:
    """Build FFN module from config."""

    ffn_type = config.architecture.ffn_type
    d_model  = config.model.d_model
    use_bias = config.architecture.bias

    ffn_dim = compute_ffn_dim(d_model, ffn_type, multiple_of=64)

    if ffn_type == "gelu_mlp":
        return GELUMLP(config, d_model, ffn_dim, use_bias)

    if ffn_type == "geglu":
        return GEGLU(config, d_model, ffn_dim, use_bias)

    if ffn_type == "swiglu":
        return SwiGLU(config, d_model, ffn_dim, use_bias)

    if ffn_type == "moe":
        raise NotImplementedError("MoE FFN is not yet implemented.")

    raise ValueError(f"Unknown FFN type: '{ffn_type}'")
