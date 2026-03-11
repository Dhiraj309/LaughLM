from flax import linen as nn

class Residual(nn.Module):
    """
    Standard transformer residual connection.
    """

    def __call__(self, x, y):
        return x + y


class ScaledResidual(nn.Module):
    """
    Residual with constant scaling.
    """

    scale: float

    def __call__(self, x, y):
        return x + self.scale * y


class DeepNormResidual(nn.Module):
    """
    DeepNorm residual scaling.

    α = (2 * N)^(1/4)
    where N is number of layers.
    """

    scale: float

    def __call__(self, x, y):
        return x + self.scale * y


def build_residual(config):

    residual_type = config.architecture.residual
    layers = config.model.num_layers

    if residual_type == "standard":
        return Residual()

    if residual_type == "scaled":
        return ScaledResidual(scale=0.5)

    if residual_type == "deep_norm":

        alpha = (2 * layers) ** 0.25

        return DeepNormResidual(scale=alpha)

    raise ValueError(f"Unknown residual type: {residual_type}")
