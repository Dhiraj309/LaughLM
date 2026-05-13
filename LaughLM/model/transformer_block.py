"""
LaughLM/model/transformer_block.py

Transformer block for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Remat fix — old code used nn.remat(self._forward) inside __call__
   which doesn't work in Flax linen (remat must wrap the Module class,
   not a method). Now remat is applied externally via build_block().

2. Parallel attention+MLP — GPT-J/PaLM style:
     out = x + Attn(Norm(x)) + MLP(Norm(x))
   instead of serial:
     out = x + MLP(Norm(x + Attn(Norm(x))))
   Saves one all-reduce in tensor-parallel and enables better
   hardware pipelining. Controlled by config.architecture.parallel_block.

3. Proper remat policy — reads from config.spmd.remat and applies
   jax.checkpoint_policies.* accordingly. Applied by build_block().

4. KV cache threading — passes kv_cache through attention for inference.

5. Signature change — __call__ now returns (output, new_cache) tuple
   to support autoregressive decoding.

References:
  GPT-J parallel: Black et al. "GPT-J-6B" (EleutherAI, 2021)
  PaLM parallel:  Chowdhery et al. (2022) — uses parallel formulation
  Remat:          MaxText → layers.py (nn.remat on Module class)
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.attention import build_attention, KVCache
from LaughLM.model.layers.mlp import build_mlp
from LaughLM.model.layers.residual import build_residual


# ────────────────────────────────────────────────────────────────
# Remat policy mapping
# ────────────────────────────────────────────────────────────────

_REMAT_POLICY_MAP = {
    "nothing_saveable":                 jax.checkpoint_policies.nothing_saveable,
    "dots_saveable":                    jax.checkpoint_policies.dots_saveable,
    "dots_with_no_batch_dims_saveable": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    "everything_saveable":              jax.checkpoint_policies.everything_saveable,
}


def get_remat_policy(policy_name: str):
    """Map config string to jax.checkpoint_policies callable."""
    if policy_name not in _REMAT_POLICY_MAP:
        raise ValueError(
            f"Unknown remat policy: '{policy_name}'. "
            f"Valid: {list(_REMAT_POLICY_MAP.keys())}"
        )
    return _REMAT_POLICY_MAP[policy_name]


# ────────────────────────────────────────────────────────────────
# Transformer Block
# ────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single transformer block supporting:
    - Serial mode (standard): Norm → Attn → Residual → Norm → MLP → Residual
    - Parallel mode (GPT-J):  out = x + Attn(Norm(x)) + MLP(Norm(x))
    - Pre/Post/Sandwich normalization placement
    - KV cache for autoregressive inference
    """

    config: LaughLMConfig

    def setup(self):
        self.norm1 = build_normalization(self.config)
        self.norm2 = build_normalization(self.config)

        self.attn = build_attention(self.config)
        self.mlp = build_mlp(self.config)

        self.residual1 = build_residual(self.config)
        self.residual2 = build_residual(self.config)

        self.norm_placement = self.config.architecture.norm_placement
        self.parallel_block = self.config.architecture.parallel_block

    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple] = None,
        doc_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        """
        Parameters
        ----------
        x          : (B, T, D) input activations
        rope_tables: (sin, cos) for RoPE
        doc_ids    : (B, T) document segment IDs
        kv_cache   : optional KVCache for inference

        Returns
        -------
        output     : (B, T, D)
        new_cache  : updated KVCache or None
        """

        # ── Parallel block (GPT-J / PaLM style) ──────────────
        if self.parallel_block:
            return self._parallel_forward(x, rope_tables, doc_ids, kv_cache)

        # ── Serial block (standard) ──────────────────────────
        return self._serial_forward(x, rope_tables, doc_ids, kv_cache)

    def _parallel_forward(self, x, rope_tables, doc_ids, kv_cache):
        """
        Parallel attention + MLP (GPT-J style):
            h = Norm(x)
            out = x + Attn(h) + MLP(h)

        Only uses pre-norm placement. Single norm applied once,
        then both attention and MLP branches read from the same h.
        This saves one sequential dependency and one all-reduce in TP.
        """
        h = self.norm1(x)

        attn_out, new_cache = self.attn(
            h, rope_tables=rope_tables, doc_ids=doc_ids, kv_cache=kv_cache
        )
        mlp_out = self.mlp(h)

        # Both branches add to residual
        out = self.residual1(x, attn_out + mlp_out)

        return out, new_cache

    def _serial_forward(self, x, rope_tables, doc_ids, kv_cache):
        """Standard serial attention → MLP with configurable norm placement."""

        if self.norm_placement == "pre":
            attn_out, new_cache = self.attn(
                self.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids, kv_cache=kv_cache
            )
            x = self.residual1(x, attn_out)

            x = self.residual2(x, self.mlp(self.norm2(x)))

            return x, new_cache

        if self.norm_placement == "post":
            attn_out, new_cache = self.attn(
                x, rope_tables=rope_tables, doc_ids=doc_ids, kv_cache=kv_cache
            )
            x = self.norm1(self.residual1(x, attn_out))

            x = self.norm2(self.residual2(x, self.mlp(x)))

            return x, new_cache

        if self.norm_placement == "sandwich":
            attn_out, new_cache = self.attn(
                self.norm1(x), rope_tables=rope_tables, doc_ids=doc_ids, kv_cache=kv_cache
            )
            x = self.residual1(x, attn_out)
            x = self.norm2(x)

            x = self.residual2(x, self.mlp(x))

            return x, new_cache

        raise ValueError(f"Unknown norm_placement: {self.norm_placement}")


# ────────────────────────────────────────────────────────────────
# Factory: build block with remat applied correctly
# ────────────────────────────────────────────────────────────────

def build_block(config: LaughLMConfig) -> nn.Module:
    """
    Build a TransformerBlock with optional remat wrapping.

    Remat is applied to the MODULE CLASS (not a method), which is
    the correct Flax linen pattern. The old code tried
    nn.remat(self._forward) which silently fails.

    Reference: MaxText → layers.py applies nn.remat(DecoderLayer)
    """
    remat_cfg = config.spmd.remat

    if remat_cfg.policy == "everything_saveable":
        # No remat — save everything (max memory, zero recompute)
        return TransformerBlock(config=config)

    # Apply remat with the specified policy
    policy = get_remat_policy(remat_cfg.policy)

    RematBlock = nn.remat(
        TransformerBlock,
        policy=policy,
        prevent_cse=remat_cfg.prevent_cse,
    )

    return RematBlock(config=config)