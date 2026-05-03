import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.utils.dtype import get_dtype


# ------------------------------------------------------------
# Stability: activation clamp (prevents bf16 overflow)
# ------------------------------------------------------------

def clamp(x: jnp.ndarray, limit: float = 30.0) -> jnp.ndarray:
    return jnp.clip(x, -limit, limit)


# ------------------------------------------------------------
# Activations (optimized JAX versions)
# ------------------------------------------------------------

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=True)


def swish(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.silu(x)


# ------------------------------------------------------------
# FFN: Standard GELU MLP
# ------------------------------------------------------------

class GELUMLP(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype   = get_dtype(self.config.parallelism.param_dtype)

        x = nn.Dense(
            self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        x = gelu(clamp(x))

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        return x


# ------------------------------------------------------------
# FFN: GEGLU
# ------------------------------------------------------------

class GEGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype   = get_dtype(self.config.parallelism.param_dtype)

        proj = nn.Dense(
            2 * self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        gate, value = jnp.split(proj, 2, axis=-1)

        gate = clamp(gate)
        x = gelu(gate) * value

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        return x


# ------------------------------------------------------------
# FFN: SwiGLU
# ------------------------------------------------------------

class SwiGLU(nn.Module):
    config: LaughLMConfig
    d_model: int
    ffn_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype   = get_dtype(self.config.parallelism.param_dtype)

        proj = nn.Dense(
            2 * self.ffn_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        gate, value = jnp.split(proj, 2, axis=-1)

        gate = clamp(gate)
        x = swish(gate) * value

        x = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            precision=jax.lax.Precision.HIGH,
        )(x)

        return x


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def compute_ffn_dim(d_model: int, ffn_type: str, multiple_of: int = 64) -> int:

    if ffn_type in ("swiglu", "geglu"):
        raw_dim = int(8 / 3 * d_model)
    else:
        raw_dim = 4 * d_model

    aligned = ((raw_dim + multiple_of - 1) // multiple_of) * multiple_of
    return aligned


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_mlp(config: LaughLMConfig) -> nn.Module:

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
