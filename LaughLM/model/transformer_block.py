
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention
from LaughLM.model.layers.mlp import build_mlp
from LaughLM.model.layers.residual import build_residual


class TransformerBlock(nn.Module):
    """
    Configurable transformer decoder block.

    Supports:
        - pre-norm
        - post-norm
        - sandwich norm

    Residual behavior is controlled by the residual axis.
    """

    config: LaughLMConfig

    def setup(self):

        # normalization layers
        self.norm1 = build_normalization(self.config)
        self.norm2 = build_normalization(self.config)

        # main submodules
        self.attn = build_attention(self.config)
        self.mlp = build_mlp(self.config)

        # residual connections
        self.residual1 = build_residual(self.config)
        self.residual2 = build_residual(self.config)

        # norm placement strategy
        self.norm_placement = self.config.architecture.norm_placement


    def __call__(self, x):

        # ------------------------------------------------------------
        # Pre-Norm Transformer (modern default)
        # ------------------------------------------------------------
        if self.norm_placement == "pre":

            h = self.attn(self.norm1(x))
            x = self.residual1(x, h)

            h = self.mlp(self.norm2(x))
            x = self.residual2(x, h)

            return x


        # ------------------------------------------------------------
        # Post-Norm Transformer (GPT-2 style)
        # ------------------------------------------------------------
        if self.norm_placement == "post":

            h = self.attn(x)
            x = self.norm1(self.residual1(x, h))

            h = self.mlp(x)
            x = self.norm2(self.residual2(x, h))

            return x


        # ------------------------------------------------------------
        # Sandwich Norm (research variant)
        # ------------------------------------------------------------
        if self.norm_placement == "sandwich":

            h = self.attn(self.norm1(x))
            x = self.residual1(x, h)

            x = self.norm2(x)

            h = self.mlp(x)
            x = self.residual2(x, h)

            return x


        raise ValueError(
            f"Unknown norm placement: {self.norm_placement}"
        )
