"""
LaughLM/model/llama/rmsnorm.py

Canonical Llama RMSNorm.

Design goals:
- HF-compatible parameter semantics
- deterministic numerics
- stable bf16/fp16 training
- TPU-native bf16 compute
- minimal architecture surface

Semantics match Hugging Face LlamaRMSNorm:
- parameter name: "weight"
- variance computed in float32
- output computed in configurable compute dtype
- parameters stored in fp32

Tensor shapes
--------------
Input:
    [batch, seq, hidden_size]

Output:
    [batch, seq, hidden_size]
"""

from flax import linen as nn

import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """
    Canonical Llama RMSNorm.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """

    hidden_size: int

    eps: float = 1e-6

    #
    # Frontier dtype policy
    #
    # - params: fp32
    # - compute: bf16
    #

    dtype: jnp.dtype = jnp.bfloat16

    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:

        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.hidden_size,),
            self.param_dtype,
        )

        # --------------------------------------------------
        # HF-compatible float32 RMS computation
        # --------------------------------------------------

        hidden_states_f32 = hidden_states.astype(
            jnp.float32
        )

        variance = jnp.mean(
            jnp.square(hidden_states_f32),
            axis=-1,
            keepdims=True,
        )

        hidden_states_normed = (
            hidden_states_f32
            * jax.lax.rsqrt(
                variance + self.eps
            )
        )

        # --------------------------------------------------
        # TPU-native compute dtype
        # --------------------------------------------------

        hidden_states_normed = (
            hidden_states_normed.astype(
                self.dtype
            )
        )

        # Match the actual normalized activation dtype rather than relying on
        # a module-level dtype annotation. This keeps strict JAX primitives
        # valid when a custom TPU VJP is present in the same traced step.
        weight = jnp.asarray(
            weight,
            dtype=hidden_states_normed.dtype,
        )

        return (
            hidden_states_normed * weight
        )
