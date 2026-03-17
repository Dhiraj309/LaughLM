
from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention
from LaughLM.model.layers.mlp import build_mlp
from LaughLM.model.layers.residual import build_residual


# ------------------------------------------------------------
# 🔥 PURE FUNCTION (NO BOUND SELF)
# ------------------------------------------------------------
def forward_block(module, x, rope_tables, doc_ids):

    if module.norm_placement == "pre":

        h = module.attn(
            module.norm1(x),
            rope_tables=rope_tables,
            doc_ids=doc_ids,
        )
        x = module.residual1(x, h)

        h = module.mlp(module.norm2(x))
        x = module.residual2(x, h)

        return x

    if module.norm_placement == "post":

        h = module.attn(x, rope_tables=rope_tables, doc_ids=doc_ids)
        x = module.norm1(module.residual1(x, h))

        h = module.mlp(x)
        x = module.norm2(module.residual2(x, h))

        return x

    if module.norm_placement == "sandwich":

        h = module.attn(module.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids)
        x = module.residual1(x, h)
        x = module.norm2(x)

        h = module.mlp(x)
        x = module.residual2(x, h)

        return x

    raise ValueError(f"Unknown norm_placement: {module.norm_placement}")


class TransformerBlock(nn.Module):

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

        # ✅ Wrap PURE function (module passed explicitly)
        remat_block = nn.remat(forward_block)

        return remat_block(self, x, rope_tables, doc_ids)
