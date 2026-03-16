

from flax import linen as nn
import math


class Residual(nn.Module):
    """
    Standard transformer residual connection.
    """

    def __call__(self, x, y):
        return x + y


class ScaledResidual(nn.Module):
    """
    Residual with deterministic scaling.

    Scale is derived from model depth to stabilize
    deeper transformers.
    """

    scale: float

    def __call__(self, x, y):
        return x + self.scale * y


class DeepNormResidual(nn.Module):
    """
    DeepNorm residual scaling.

    Paper:
    DeepNet: Scaling Transformers to 1,000 Layers

    α = (2 * N)^(1/4)

    Residual branch becomes:
        x + (1/α) * f(x)
    """

    scale: float

    def __call__(self, x, y):
        return x + self.scale * y


def build_residual(config):
    """
    Build residual connection module based on configuration.
    """

    residual_type = config.architecture.residual
    layers = config.model.num_layers

    if residual_type == "standard":
        return Residual()

    if residual_type == "scaled":

        scale = 1 / math.sqrt(layers)

        return ScaledResidual(scale=scale)

    if residual_type == "deep_norm":

        alpha = (2 * layers) ** 0.25
        scale = 1 / alpha

        return DeepNormResidual(scale=scale)

    raise ValueError(f"Unknown residual type: {residual_type}")
