"""
LaughLM/model/layers/attention.py

Frontier-grade attention module for decoder-only LLM pretraining.

Key design decisions
--------------------
1. BTNH layout (B, T, num_heads, head_dim) throughout — this is the native
   layout for jax.nn.dot_product_attention. Zero transposes in the hot path.

2. jax.nn.dot_product_attention with is_causal=True — dispatches to:
   - cuDNN FlashAttention on GPU (O(T) memory, tiled softmax)
   - XLA with causal mask on TPU/CPU (O(T²) memory, but mask is built-in)
   For TPU Splash Attention, override implementation at call site.

3. Native GQA — jax.nn.dot_product_attention handles Q_heads != KV_heads
   natively via broadcasting. No jnp.repeat, no jnp.broadcast_to needed.
   MQA is just GQA with num_kv_heads=1.

4. Unified class — single CausalAttention class handles MHA, MQA, and GQA
   via num_heads / num_kv_heads. Eliminates code duplication and the
   factory pattern that hid bugs across three separate classes.

5. Scale after RoPE — Q scaling is applied after rotary position encoding,
   which is numerically better because RoPE rotations should operate on
   the raw query vectors, not pre-scaled ones.

Reference implementations
-------------------------
- MaxText (Google): AI-Hypercomputer/maxtext/layers/attentions.py
- JAX API: jax/_src/nn/functions.py (dot_product_attention)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope
from LaughLM.utils.dtype import get_dtype


# ------------------------------------------------------------
# Layout helpers (BTNH — zero transposes)
# ------------------------------------------------------------

def reshape_to_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    """
    Reshape (B, T, D) → (B, T, H, head_dim).

    This is a pure reshape — no transpose. The resulting (B, T, H, D)
    layout is the native format for jax.nn.dot_product_attention and
    for apply_rope in positional.py.
    """
    b, t, d = x.shape
    head_dim = d // num_heads
    return x.reshape(b, t, num_heads, head_dim)


def reshape_from_heads(x: jnp.ndarray) -> jnp.ndarray:
    """
    Reshape (B, T, H, head_dim) → (B, T, D).

    Pure reshape — no transpose. Inverse of reshape_to_heads.
    """
    b, t, h, d = x.shape
    return x.reshape(b, t, h * d)


# ------------------------------------------------------------
# Attention core — flash / splash / XLA dispatch
# ------------------------------------------------------------

def causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    implementation: Optional[str] = None,
) -> jnp.ndarray:
    """
    Causal (autoregressive) attention with automatic kernel dispatch.

    Parameters
    ----------
    q : (B, T, N, H)  — query, N = num_query_heads
    k : (B, S, K, H)  — key,   K = num_kv_heads (K <= N, N % K == 0)
    v : (B, S, K, H)  — value
    implementation : 'xla', 'cudnn', or None (auto).
        'cudnn' → cuDNN FlashAttention (GPU, O(T) memory)
        'xla'   → XLA einsum with causal mask (TPU/CPU, O(T²) memory)
        None    → JAX auto-selects (currently defaults to XLA)

    Returns
    -------
    out : (B, T, N, H)

    Notes
    -----
    - is_causal=True avoids materializing a (T, T) mask tensor on cuDNN
    - GQA is handled natively: K < N is supported, N % K == 0 required
    - MQA is GQA with K=1
    - scale is NOT applied here — it's applied to Q before this call
    """
    return jax.nn.dot_product_attention(
        q, k, v,
        is_causal=True,
        implementation=implementation,
    )


# ------------------------------------------------------------
# Unified Causal Attention (MHA / MQA / GQA)
# ------------------------------------------------------------

class CausalAttention(nn.Module):
    """
    Unified multi-head / multi-query / grouped-query causal attention.

    When num_kv_heads == num_heads → standard MHA
    When num_kv_heads == 1        → MQA
    When 1 < num_kv_heads < num_heads → GQA

    All three use the same code path — the difference is only in the
    QKV projection dimensions. jax.nn.dot_product_attention handles
    the head broadcasting internally.
    """

    config: LaughLMConfig
    d_model: int
    num_heads: int
    num_kv_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_tables: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        doc_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        compute_dtype = get_dtype(self.config.parallelism.compute_dtype)
        param_dtype = get_dtype(self.config.parallelism.param_dtype)

        head_dim = self.d_model // self.num_heads
        scale = head_dim ** -0.5

        # Q dim = num_heads * head_dim = d_model
        # KV dim = num_kv_heads * head_dim  (smaller for GQA/MQA)
        kv_dim = self.num_kv_heads * head_dim
        qkv_dim = self.d_model + 2 * kv_dim

        # --------------------------------------------------------
        # Fused QKV projection (single matmul)
        # --------------------------------------------------------
        qkv = nn.Dense(
            qkv_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="qkv_proj",
        )(x)

        # Split into Q, K, V
        q = qkv[..., :self.d_model]
        k = qkv[..., self.d_model:self.d_model + kv_dim]
        v = qkv[..., self.d_model + kv_dim:]

        # --------------------------------------------------------
        # Reshape to (B, T, H, head_dim) — no transpose
        # --------------------------------------------------------
        q = reshape_to_heads(q, self.num_heads)         # (B, T, N, D)
        k = reshape_to_heads(k, self.num_kv_heads)      # (B, T, K, D)
        v = reshape_to_heads(v, self.num_kv_heads)      # (B, T, K, D)

        # --------------------------------------------------------
        # RoPE (applied before scaling — correct ordering)
        #
        # apply_rope expects (B, T, H, D) which is exactly our layout.
        # No transposes needed.
        # --------------------------------------------------------
        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        # --------------------------------------------------------
        # Scale Q after RoPE
        #
        # RoPE rotations should operate on raw Q vectors.
        # Scaling after preserves the rotation geometry.
        # --------------------------------------------------------
        q = q * scale

        # --------------------------------------------------------
        # Causal attention (flash/splash/XLA dispatch)
        #
        # jax.nn.dot_product_attention handles GQA natively:
        #   Q: (B, T, N, D)  — N query heads
        #   K: (B, T, K, D)  — K kv heads, N % K == 0
        #   V: (B, T, K, D)
        # No repeat / broadcast_to needed.
        # --------------------------------------------------------
        out = causal_attention(q, k, v)    # (B, T, N, D)

        # --------------------------------------------------------
        # Merge heads: (B, T, N, D) → (B, T, d_model) — no transpose
        # --------------------------------------------------------
        out = reshape_from_heads(out)

        # --------------------------------------------------------
        # Output projection
        # --------------------------------------------------------
        out = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="out_proj",
        )(out)

        return out


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_attention(config: LaughLMConfig) -> CausalAttention:
    """
    Build attention module from config.

    All variants (MHA, MQA, GQA) use the same CausalAttention class.
    The difference is only in num_kv_heads:
        MHA: num_kv_heads = num_heads
        MQA: num_kv_heads = 1
        GQA: num_kv_heads = config.model.num_kv_heads
    """

    variant = config.architecture.attention_variant
    num_heads = config.model.num_heads

    if variant == "mha":
        num_kv_heads = num_heads

    elif variant == "mqa":
        num_kv_heads = 1

    elif variant == "gqa":
        num_kv_heads = config.model.num_kv_heads
        if num_kv_heads is None:
            raise ValueError(
                f"attention_variant='gqa' requires model.num_kv_heads to be set. "
                f"Typical values for {num_heads} heads: "
                f"{num_heads // 4} (4:1) or {num_heads // 8} (8:1)."
            )

    else:
        raise ValueError(f"Unknown attention variant: '{variant}'")

    return CausalAttention(
        config=config,
        d_model=config.model.d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_bias=config.architecture.bias,
    )
