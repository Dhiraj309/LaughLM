"""
LaughLM/model/llama/decoder.py

Canonical Llama decoder layer.

Design goals:
- HF-compatible semantics
- deterministic residual ordering
- explicit architecture structure
- minimal abstraction surface

Tensor conventions
------------------
hidden_states:
    [B, T, D]

attention_mask:
    [B, 1, T_q, T_kv]

positions:
    [B, T]
"""

from typing import Optional

from flax import linen as nn
import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig
from LaughLM.model.llama.rmsnorm import RMSNorm
from LaughLM.model.llama.attention import LlamaAttention
from LaughLM.model.llama.mlp import LlamaMLP
from LaughLM.model.llama.kv_cache import KVCache


class LlamaDecoderLayer(nn.Module):

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        positions: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        mode: str = "train",
    ) -> tuple[
        jnp.ndarray,
        Optional[KVCache],
    ]:
        """
        Parameters
        ----------
        hidden_states:
            [B, T, D]

        positions:
            [B, T]

        attention_mask:
            [B, 1, T_q, T_kv]

        mode:
            "train"
            "prefill"
            "decode"
        """

        config = self.config

        # ──────────────────────────────────────────
        # Attention block
        # ──────────────────────────────────────────

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            name="input_layernorm",
        )(hidden_states)

        hidden_states, updated_cache = (
            LlamaAttention(
                config=config,
                name="self_attn",
            )(
                hidden_states=hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                mode=mode,
            )
        )

        hidden_states = residual + hidden_states

        # ──────────────────────────────────────────
        # MLP block
        # ──────────────────────────────────────────

        residual = hidden_states

        hidden_states = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            name="post_attention_layernorm",
        )(hidden_states)

        hidden_states = LlamaMLP(
            config=config,
            name="mlp",
        )(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, updated_cache