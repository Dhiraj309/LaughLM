"""
LaughLM/model/llama/rmsnorm.py

Canonical Llama RMSNorm.

Design goals:
- HF-compatible parameter semantics
- deterministic numerics
- stable bf16/fp16 training
- minimal architecture surface

Semantics match Hugging Face LlamaRMSNorm:
- parameter name: "weight"
- variance computed in float32
- output cast back to input dtype

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

    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:

        input_dtype = hidden_states.dtype

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
        # Cast back to activation dtype
        # --------------------------------------------------

        hidden_states_normed = (
            hidden_states_normed.astype(
                input_dtype
            )
        )

        weight = weight.astype(
            input_dtype
        )

        return (
            hidden_states_normed * weight
        )
