"""
LaughLM/model/llama/mlp.py

Canonical Llama SwiGLU MLP.

Design goals:
- HF-compatible parameter naming
- deterministic semantics
- minimal architecture surface
- stable bf16/fp16 behavior

Tensor conventions
------------------
Input:
    [B, T, D]

Output:
    [B, T, D]
"""

from flax import linen as nn

import jax
import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig


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

        gate_proj = nn.Dense(
            config.intermediate_size,
            use_bias=config.mlp_bias,
            name="gate_proj",
        )

        up_proj = nn.Dense(
            config.intermediate_size,
            use_bias=config.mlp_bias,
            name="up_proj",
        )

        down_proj = nn.Dense(
            config.hidden_size,
            use_bias=config.mlp_bias,
            name="down_proj",
        )

        gate = gate_proj(hidden_states)

        gate = jax.nn.silu(gate)

        up = up_proj(hidden_states)

        hidden_states = gate * up

        hidden_states = down_proj(hidden_states)

        return hidden_states