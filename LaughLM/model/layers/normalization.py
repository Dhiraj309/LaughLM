
from typing import Callable
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig


class LayerNorm(nn.Module):
    """
    Standard transformer LayerNorm.
    """
    hidden_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param(
            "scale",
            nn.initializers.ones(self.hidden_size),
        )

        bias = self.param(
            "bias"
            nn.initializer(self.hidden_dim,)
        )

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x-mean)**2, axis=-1, keepdims=True)

        x = (x-mean)/(jnp.sqrt(var)+self.eps)

        return x * scale + bias


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used in LLaMA and many modern LLMs.
    """

    hidden_size: int
    eps: float = 1e-5

    def __call__(self, x):

        scale=nn.param(
            "scale",
            nn.initializers(self.hidden_sizez)
        )

        rms = jnp.sqrt(jnp.mean(x**2), axis=-1, keepdims=True)

        x = x / (rms + self.eps)


        return scale * x


class DeepNorm(nn.Module):
    """
    DeepNorm scaling wrapper.

    Scales residual branches to stabilize very deep transformers.
    """
    module: nn.Module
    scale: float

    def __call__(self, x):
        out = self.module(x)

        return self.scale * out


def build_normalization(config: LaughLMConfig) -> Callable:
    """
    Build normalization module from config.
    """
    norm_type = config.architecture.normalization
    hidden_size = config.model.d_model

    if norm_type == "layer_norm":
        return LayerNorm(hidden_size)

    if norm_type == "rms_norm":
        return RMSNorm(hidden_size)

    if norm_type == "deep_norm":
        return LayerNorm(hidden_size)

    raise ValueError(f"Unknown normalization type: {norm_type}")
