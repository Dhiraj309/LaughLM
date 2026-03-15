from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention
from LaughLM.model.layers.mlp import build_mlp
from LaughLM.model.layers.residual import build_residual


class TransformerBlock(nn.Module):
    """
    Configurable transformer decoder block.

    Supports three norm placements:
        pre      — normalize BEFORE attention/FFN (modern default, recommended)
        post     — normalize AFTER residual (original Transformer, GPT-2)
        sandwich — normalize before and after each sub-layer

    RoPE threading:
        rope_tables = (sin, cos) is passed in from GPTModel and forwarded
        to the attention layer. When None, attention uses no positional bias
        (e.g. for learned/sinusoidal embeddings added at the input).

    Cross-document masking:
        doc_ids is passed through to attention's causal mask builder.
        Required when sequence packing is enabled (config.data.packing=True).
    """

    config: LaughLMConfig

    def setup(self):

        self.norm1 = build_normalization(self.config)
        self.norm2 = build_normalization(self.config)

        self.attn = build_attention(self.config)
        self.mlp  = build_mlp(self.config)

        self.residual1 = build_residual(self.config)
        self.residual2 = build_residual(self.config)

        self.norm_placement = self.config.architecture.norm_placement

    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        x           : [B, T, D] input hidden states
        rope_tables : (sin, cos) from build_rope_tables(), or None
        doc_ids     : [B, T] integer document IDs for cross-doc masking, or None

        Returns
        -------
        x : [B, T, D] output hidden states
        """

        # ----------------------------------------------------------------
        # Pre-Norm (modern default — Llama, DeepSeek, Mistral, all 2024 models)
        # ----------------------------------------------------------------
        if self.norm_placement == "pre":

            # Attention sub-layer
            h = self.attn(
                self.norm1(x),
                rope_tables=rope_tables,
                doc_ids=doc_ids,
            )
            x = self.residual1(x, h)

            # FFN sub-layer
            h = self.mlp(self.norm2(x))
            x = self.residual2(x, h)

            return x

        # ----------------------------------------------------------------
        # Post-Norm (GPT-2 style — less stable, not recommended)
        # ----------------------------------------------------------------
        if self.norm_placement == "post":

            h = self.attn(x, rope_tables=rope_tables, doc_ids=doc_ids)
            x = self.norm1(self.residual1(x, h))

            h = self.mlp(x)
            x = self.norm2(self.residual2(x, h))

            return x

        # ----------------------------------------------------------------
        # Sandwich Norm (research variant)
        # ----------------------------------------------------------------
        if self.norm_placement == "sandwich":

            h = self.attn(self.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids)
            x = self.residual1(x, h)
            x = self.norm2(x)

            h = self.mlp(x)
            x = self.residual2(x, h)

            return x

        raise ValueError(
            f"Unknown norm_placement: '{self.norm_placement}'. "
            f"Valid options: pre, post, sandwich."
        )
