"""
LaughLM/model/llama/mlp.py

Canonical Llama SwiGLU MLP.

Frontier-grade SPMD additions:
────────────────────────────────────────────────────
1. Logical activation constraints
2. Tensor-parallel-aware MLP sharding
3. Stable bf16 SwiGLU execution
4. GSPMD-compatible activation layouts
5. Explicit intermediate-axis semantics
6. Sequence-parallel-ready activations

Design goals:
- HF-compatible parameter naming
- deterministic semantics
- HF-compatible initialization
- minimal architecture surface
- stable bf16/fp16 behavior
- tensor-parallel compatible layouts

Tensor conventions
──────────────────
Input:
    [B, T, D]

Intermediate:
    [B, T, I]

Output:
    [B, T, D]
"""

import jax
import jax.numpy as jnp

from flax import linen as nn

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.initialization import (
    create_dense,
    constrain_hidden_states,
    constrain_mlp_intermediate,
)


class LlamaMLP(nn.Module):

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        hidden_states:
            [B, T, D]

        Returns
        -------
        hidden_states:
            [B, T, D]
        """

        config = self.config

        # --------------------------------------------------
        # Input activation constraint
        # --------------------------------------------------

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        # --------------------------------------------------
        # SwiGLU projections
        # --------------------------------------------------

        gate_proj = create_dense(
            features=(
                config.intermediate_size
            ),
            config=config,
            use_bias=config.mlp_bias,
            name="gate_proj",
        )

        up_proj = create_dense(
            features=(
                config.intermediate_size
            ),
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

        # --------------------------------------------------
        # Gate projection
        # --------------------------------------------------

        gate = gate_proj(
            hidden_states
        )

        gate = constrain_mlp_intermediate(
            gate
        )

        # --------------------------------------------------
        # SwiGLU activation
        # --------------------------------------------------
        #
        # Frontier standard:
        #   silu in compute dtype
        #   accumulation internally promoted by XLA
        #

        gate = jax.nn.silu(
            gate
        )

        # --------------------------------------------------
        # Up projection
        # --------------------------------------------------

        up = up_proj(
            hidden_states
        )

        up = constrain_mlp_intermediate(
            up
        )

        # --------------------------------------------------
        # SwiGLU fusion
        # --------------------------------------------------

        hidden_states = (
            gate * up
        )

        hidden_states = constrain_mlp_intermediate(
            hidden_states
        )

        # --------------------------------------------------
        # Down projection
        # --------------------------------------------------

        hidden_states = down_proj(
            hidden_states
        )

        # --------------------------------------------------
        # Output activation constraint
        # --------------------------------------------------

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        return hidden_states
