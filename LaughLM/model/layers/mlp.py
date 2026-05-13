"""
LaughLM/model/layers/mlp.py

Feed-forward network layers.

Fixes:
- FFN dim aligned to 128 (v5e MXU tile size), was 64
- Removed clamp() — unnecessary overhead with proper init
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.utils.dtype import resolve_compute_dtype, resolve_param_dtype


def gelu(x):
    return jax.nn.gelu(x, approximate=True)

def swish(x):
    return jax.nn.silu(x)


class GELUMLP(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        x = nn.Dense(self.ffn_dim, use_bias=self.use_bias,
                     dtype=compute_dtype, param_dtype=param_dtype)(x)
        x = gelu(x)
        x = nn.Dense(self.d_model, use_bias=self.use_bias,
                     dtype=compute_dtype, param_dtype=param_dtype)(x)
        return x


class GEGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        proj = nn.Dense(2 * self.ffn_dim, use_bias=self.use_bias,
                        dtype=compute_dtype, param_dtype=param_dtype)(x)
        gate, value = jnp.split(proj, 2, axis=-1)
        x = gelu(gate) * value
        x = nn.Dense(self.d_model, use_bias=self.use_bias,
                     dtype=compute_dtype, param_dtype=param_dtype)(x)
        return x


class SwiGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        proj = nn.Dense(2 * self.ffn_dim, use_bias=self.use_bias,
                        dtype=compute_dtype, param_dtype=param_dtype)(x)
        gate, value = jnp.split(proj, 2, axis=-1)
        x = swish(gate) * value
        x = nn.Dense(self.d_model, use_bias=self.use_bias,
                     dtype=compute_dtype, param_dtype=param_dtype)(x)
        return x


def compute_ffn_dim(d_model: int, ffn_type: str, multiple_of: int = 128) -> int:
    """
    Compute FFN intermediate dimension.

    Aligned to 128 (TPU v5e MXU tile size) for zero padding waste.
    SwiGLU/GEGLU: 8/3 × d_model. GELU: 4 × d_model.
    """
    if ffn_type in ("swiglu", "geglu"):
        raw_dim = int(8 / 3 * d_model)
    else:
        raw_dim = 4 * d_model

    aligned = ((raw_dim + multiple_of - 1) // multiple_of) * multiple_of
    return aligned


def build_mlp(config: LaughLMConfig) -> nn.Module:
    ffn_type = config.architecture.ffn_type
    d_model = config.model.d_model
    use_bias = config.architecture.bias

    ffn_dim = compute_ffn_dim(d_model, ffn_type, multiple_of=128)

    if ffn_type == "gelu_mlp":
        return GELUMLP(config, d_model, ffn_dim, use_bias)
    if ffn_type == "geglu":
        return GEGLU(config, d_model, ffn_dim, use_bias)
    if ffn_type == "swiglu":
        return SwiGLU(config, d_model, ffn_dim, use_bias)
    if ffn_type == "moe":
        raise NotImplementedError("MoE not implemented.")
    raise ValueError(f"Unknown FFN type: '{ffn_type}'")