
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention
from LaughLM.model.layers.mlp import build_mlp

class TransformerBlock(nn.Module):
    """
    Configurable transformer decoder block.
    """
    config: LaughLMConfig

    def setup(self):
        self.norm1 = build_normalization(self.config)
        self.norm2 = build_normalization(self.config)

        self.attn = build_attention(self.config)
        self.mlp = build_mlp(self.config)

        self.norm_placement = self.config.architecture.norm_placement


    def __call__(self, x):
        if self.norm_placement == "pre":
            h = self.attn(self.norm1(x))
            x = x + h

            h = self.mlp(self.norm2(x))
            x = x + h

            return x

        if self.norm_placement == "post":
            h = self.attn(x)
            x = self.norm1(x+h)

            h = self.mlp(x)
            x = self.norm2(x+h)

            return x

        if self.norm_placement == "sandwich":
            h = self.attn(self.norm1(x))
            x = x + h

            x = self.norm2(x)

            h = self.mlp(h)

            x = x + h

            return x

        return ValueError(f"Unknown norm placement: {self.norm_placement}")
