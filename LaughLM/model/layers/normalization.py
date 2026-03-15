
import jax.numpy as jnp
from flax import linen as nn

class LayerNorm(nn.Module):
    """
    Standard LayerNorm used in GPT-style transformers.
    """

    hidden_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):

        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.hidden_size,),
        )

        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (self.hidden_size,),
        )

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)

        x = (x - mean) / jnp.sqrt(var + self.eps)

        return x * scale + bias

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization used in LLaMA.
    """

    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):

        scale = self.param(
            "scale",
            nn.initializers.ones,
            (self.hidden_size,),
        )

        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True))

        x = x / (rms + self.eps)

        return scale * x


def build_normalization(config):
    """
    Build normalization module from config.
    """

    norm_type = config.architecture.normalization
    hidden = config.model.d_model

    if norm_type == "layer_norm":
        return LayerNorm(hidden)

    if norm_type == "rms_norm":
        return RMSNorm(hidden)

    if norm_type == "deep_norm":
        # DeepNorm uses LayerNorm + residual scaling
        return LayerNorm(hidden)

    raise ValueError(f"Unknown normalization type: {norm_type}")
