"""
LaughLM/model/llama/mlp.py

Canonical Llama SwiGLU MLP.

Design goals:
- HF-compatible parameter naming
- deterministic semantics
- HF-compatible initialization
- minimal architecture surface
- stable bf16/fp16 behavior

Tensor conventions
------------------
Input:
    [B, T, D]

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
        # SwiGLU
        # --------------------------------------------------

        gate = gate_proj(
            hidden_states
        )

        gate = jax.nn.silu(
            gate
        )

        up = up_proj(
            hidden_states
        )

        hidden_states = (
            gate * up
        )

        hidden_states = down_proj(
            hidden_states
        )

        return hidden_states
