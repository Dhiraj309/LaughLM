"""
LaughLM/model/llama/decoder.py

Canonical Llama decoder layer.

Frontier-grade SPMD additions:
────────────────────────────────────────────────────
1. Logical activation constraints
2. Tensor-parallel-safe residual joins
3. Parallel-block compatibility hooks
4. Remat-safe structure
5. Stable bf16 residual semantics
6. Sequence-parallel-ready layouts
7. Communication-minimized activation flow

Design goals:
- HF-compatible semantics
- deterministic residual ordering
- explicit architecture structure
- minimal abstraction surface
- tensor-parallel compatibility
- GSPMD-safe residual structure

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

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.rmsnorm import (
    RMSNorm,
)

from LaughLM.model.llama.attention import (
    LlamaAttention,
)

from LaughLM.model.llama.mlp import (
    LlamaMLP,
)

from LaughLM.model.llama.kv_cache import (
    KVCache,
)

from LaughLM.model.llama.initialization import (
    constrain_hidden_states,
)


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

        # --------------------------------------------------
        # Input layout constraint
        # --------------------------------------------------

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        # ==================================================
        # Standard serial residual path
        # ==================================================

        if not getattr(
            config,
            "parallel_block",
            False,
        ):

            # ──────────────────────────────────────────
            # Attention block
            # ──────────────────────────────────────────

            residual = hidden_states

            hidden_states = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                name="input_layernorm",
            )(hidden_states)

            hidden_states = constrain_hidden_states(
                hidden_states
            )

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

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            # ------------------------------------------
            # Residual join
            # ------------------------------------------

            hidden_states = (
                residual + hidden_states
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            # ──────────────────────────────────────────
            # MLP block
            # ──────────────────────────────────────────

            residual = hidden_states

            hidden_states = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                name="post_attention_layernorm",
            )(hidden_states)

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            hidden_states = LlamaMLP(
                config=config,
                name="mlp",
            )(hidden_states)

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            # ------------------------------------------
            # Residual join
            # ------------------------------------------

            hidden_states = (
                residual + hidden_states
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            return (
                hidden_states,
                updated_cache,
            )

        # ==================================================
        # Parallel block path (PaLM / GPT-J / MPT)
        # ==================================================
        #
        # out =
        #   x
        #   + Attn(Norm(x))
        #   + MLP(Norm(x))
        #
        # Advantages:
        # - fewer synchronization points
        # - better tensor-parallel overlap
        # - reduced pipeline bubbles
        #
        # Used in:
        # - PaLM
        # - GPT-J
        # - MPT
        #

        residual = hidden_states

        normed_hidden = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            name="input_layernorm",
        )(hidden_states)

        normed_hidden = constrain_hidden_states(
            normed_hidden
        )

        # --------------------------------------------------
        # Attention branch
        # --------------------------------------------------

        attn_output, updated_cache = (
            LlamaAttention(
                config=config,
                name="self_attn",
            )(
                hidden_states=normed_hidden,
                positions=positions,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                mode=mode,
            )
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        # --------------------------------------------------
        # MLP branch
        # --------------------------------------------------

        mlp_output = LlamaMLP(
            config=config,
            name="mlp",
        )(
            normed_hidden
        )

        mlp_output = constrain_hidden_states(
            mlp_output
        )

        # --------------------------------------------------
        # Parallel residual merge
        # --------------------------------------------------

        hidden_states = (
            residual
            + attn_output
            + mlp_output
        )

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        return (
            hidden_states,
            updated_cache,
        )
