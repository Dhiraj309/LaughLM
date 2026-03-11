import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig


def gelu(x):
    """
    Gaussian Error Linear Unit.
    Used in GPT-2 / BERT.
    """
    return 0.5 * x * (1 + jnp.tanh(
        jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)
    ))


def swish(x):
    """
    Swish activation used in SwiGLU
    """
    return x * nn.sigmoid(x)


class GELUMLP(nn.Module):
    d_model: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.ffn_dim)(x)
        x = gelu(x)
        x = nn.Dense(self.d_model)(x)

        return x


class GEGLU(nn.Module):
    d_model: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.ffn_dim)(x)
        value = nn.Dense(self.ffn_dim)(x)

        x = gelu(gate) * value

        x = nn.Dense(self.d_model)(x)

        return x


class SwiGLU(nn.Module):
    d_model: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.ffn_dim)(x)
        value = nn.Dense(self.ffn_dim)(x)

        x = swish(gate) * value

        x = nn.Dense(self.d_model)(x)

        return x


def build_mlp(config: LaughLMConfig):

    ffn_type = config.architecture.ffn_type
    d_model = config.model.d_model

    ffn_dim = 4 * d_model

    if ffn_type == "gelu_mlp":
        return GELUMLP(d_model, ffn_dim)

    if ffn_type == "geglu":
        return GEGLU(d_model, ffn_dim)

    if ffn_type == "swiglu":
        return SwiGLU(d_model, ffn_dim)

    if ffn_type == "moe":
        raise NotImplementedError("MoE will be added later")

    raise ValueError(f"Unknown FFN type: {ffn_type}")
