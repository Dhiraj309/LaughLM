from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention
from LaughLM.model.layers.mlp import build_mlp
from LaughLM.model.layers.residual import build_residual


class TransformerBlock(nn.Module):

    config: LaughLMConfig

    def setup(self):

        self.norm1 = build_normalization(self.config)
        self.norm2 = build_normalization(self.config)

        self.attn = build_attention(self.config)
        self.mlp = build_mlp(self.config)

        self.residual1 = build_residual(self.config)
        self.residual2 = build_residual(self.config)

        self.norm_placement = self.config.architecture.norm_placement
        self.use_remat = getattr(self.config.runtime, "remat_blocks", False)

    def _forward(self, x, rope_tables, doc_ids):

        # ------------------------------------------------------------
        # PRE-NORM (best for fusion)
        # ------------------------------------------------------------
        if self.norm_placement == "pre":
            x = self.residual1(
                x,
                self.attn(self.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids),
            )
            x = self.residual2(
                x,
                self.mlp(self.norm2(x)),
            )
            return x

        # ------------------------------------------------------------
        # POST-NORM
        # ------------------------------------------------------------
        if self.norm_placement == "post":
            x = self.norm1(
                self.residual1(
                    x,
                    self.attn(x, rope_tables=rope_tables, doc_ids=doc_ids),
                )
            )
            x = self.norm2(
                self.residual2(
                    x,
                    self.mlp(x),
                )
            )
            return x

        # ------------------------------------------------------------
        # SANDWICH
        # ------------------------------------------------------------
        if self.norm_placement == "sandwich":
            x = self.residual1(
                x,
                self.attn(self.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids),
            )
            x = self.norm2(x)
            x = self.residual2(
                x,
                self.mlp(x),
            )
            return x

        raise ValueError(f"Unknown norm_placement: {self.norm_placement}")

    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        if self.use_remat:
            return nn.remat(self._forward)(x, rope_tables, doc_ids)
        else:
            return self._forward(x, rope_tables, doc_ids)
