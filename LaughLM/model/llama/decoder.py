"""
LaughLM/model/llama/decoder.py

Canonical Llama decoder layer.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from flax import linen as nn

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

from LaughLM.model.llama.remat import (
    maybe_remat,
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

        config = self.config

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        # ====================================================
        # Selective remat modules
        # ====================================================

        AttentionModule = maybe_remat(
            LlamaAttention,
            enabled=config.remat_attention,
            policy=config.remat_policy,
            prevent_cse=config.prevent_cse,
        )

        MLPModule = maybe_remat(
            LlamaMLP,
            enabled=config.remat_mlp,
            policy=config.remat_policy,
            prevent_cse=config.prevent_cse,
        )

        # ====================================================
        # Standard serial block
        # ====================================================

        if not config.parallel_block:

            # ------------------------------------------------
            # Attention block
            # ------------------------------------------------

            residual = hidden_states

            hidden_states = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.compute_dtype,
                param_dtype=config.param_dtype,
                name="input_layernorm",
            )(
                hidden_states
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            (
                hidden_states,
                updated_cache,
            ) = AttentionModule(
                config=config,
                name="self_attn",
            )(
                hidden_states=hidden_states,
                positions=positions,

                #
                # Runtime attention builds masks internally.
                #
                attention_mask=None,

                kv_cache=kv_cache,
                mode=mode,
            )

            hidden_states = (
                residual + hidden_states
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            # ------------------------------------------------
            # MLP block
            # ------------------------------------------------

            residual = hidden_states

            hidden_states = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.compute_dtype,
                param_dtype=config.param_dtype,
                name="post_attention_layernorm",
            )(
                hidden_states
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            hidden_states = MLPModule(
                config=config,
                name="mlp",
            )(
                hidden_states
            )

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

        # ====================================================
        # Parallel block
        # ====================================================

        residual = hidden_states

        normed_hidden = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.compute_dtype,
            param_dtype=config.param_dtype,
            name="input_layernorm",
        )(
            hidden_states
        )

        normed_hidden = constrain_hidden_states(
            normed_hidden
        )

        # ----------------------------------------------------
        # Attention branch
        # ----------------------------------------------------

        (
            attn_output,
            updated_cache,
        ) = AttentionModule(
            config=config,
            name="self_attn",
        )(
            hidden_states=normed_hidden,
            positions=positions,

            #
            # Runtime attention builds masks internally.
            #
            attention_mask=None,

            kv_cache=kv_cache,
            mode=mode,
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        # ----------------------------------------------------
        # MLP branch
        # ----------------------------------------------------

        mlp_output = MLPModule(
            config=config,
            name="mlp",
        )(
            normed_hidden
        )

        mlp_output = constrain_hidden_states(
            mlp_output
        )

        # ----------------------------------------------------
        # Residual merge
        # ----------------------------------------------------

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
