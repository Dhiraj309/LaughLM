"""
LaughLM/model/llama/mlp.py

Canonical Llama SwiGLU MLP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flax import linen as nn

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.initialization import (
    create_dense,
    constrain_hidden_states,
    constrain_mlp_activations,
)


class LlamaMLP(nn.Module):

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Input:
            [B, T, D]

        Output:
            [B, T, D]
        """

        config = self.config

        # ====================================================
        # Input constraint
        # ====================================================

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        # ====================================================
        # Projections
        # ====================================================

        gate_proj = create_dense(
            features=config.intermediate_size,
            config=config,
            use_bias=config.mlp_bias,
            name="gate_proj",
        )

        up_proj = create_dense(
            features=config.intermediate_size,
            config=config,
            use_bias=config.mlp_bias,
            name="up_proj",
        )

        down_proj = create_dense(
            features=config.hidden_size,
            config=config,
            use_bias=config.mlp_bias,
            name="down_proj",
        )

        # ====================================================
        # Gate path
        # ====================================================

        gate = gate_proj(
            hidden_states
        )

        gate = constrain_mlp_activations(
            gate
        )

        # ----------------------------------------------------
        # SwiGLU activation
        #
        # Compute in compute_dtype.
        # ----------------------------------------------------

        gate = jax.nn.silu(
            gate
        )

        # ====================================================
        # Up path
        # ====================================================

        up = up_proj(
            hidden_states
        )

        up = constrain_mlp_activations(
            up
        )

        # ====================================================
        # SwiGLU fusion
        # ====================================================

        hidden_states = gate * up

        hidden_states = (
            constrain_mlp_activations(
                hidden_states
            )
        )

        # ====================================================
        # Down projection
        # ====================================================

        hidden_states = down_proj(
            hidden_states
        )

        # ====================================================
        # Output constraint
        # ====================================================

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        return hidden_states
