"""
LaughLM/model/layers/attention.py

Frontier-grade attention module for decoder-only LLM pretraining.

Key design decisions
--------------------
1. BTNH layout (B, T, num_heads, head_dim) throughout — this is the native
   layout for jax.nn.dot_product_attention. Zero transposes in the main path.

2. Hardware-aware kernel dispatch:
   - TPU  → Splash Attention (Pallas kernel, O(T) memory, native GQA)
   - GPU (Ampere+) → cuDNN FlashAttention via jax.nn (O(T) memory)
   - GPU (pre-Ampere, e.g. T4) → XLA with is_causal=True (O(T²) memory)
   - CPU  → XLA fallback (O(T²), for testing only)

   CRITICAL: jax.nn.dot_product_attention has NO TPU flash path.
   The 'xla' implementation materializes a full (B, N, T, T) attention
   matrix, causing OOM on TPU for real training configs. Splash Attention
   avoids this entirely via tiled on-chip computation.

   cuDNN FlashAttention requires Ampere+ (SM >= 8.0). Pre-Ampere GPUs
   (T4/SM 7.5, V100/SM 7.0) must use XLA fallback.

3. Native GQA — both Splash Attention and jax.nn.dot_product_attention
   handle Q_heads != KV_heads natively. No jnp.repeat or broadcast_to.

4. Unified class — single CausalAttention class handles MHA, MQA, and GQA
   via num_heads / num_kv_heads.

5. Scale after RoPE — Q scaling is applied after rotary position encoding.

Reference implementations
-------------------------
- MaxText (Google): AI-Hypercomputer/maxtext/layers/attention_op.py
- JAX Splash Attention: jax/experimental/pallas/ops/tpu/splash_attention/
- JAX API: jax/_src/nn/functions.py (dot_product_attention)
"""

import functools
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope
from LaughLM.utils.dtype import get_dtype


# ------------------------------------------------------------
# Layout helpers (BTNH — zero transposes for RoPE + projections)
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
# GPU capability detection (cached)
# ------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _gpu_supports_cudnn_flash() -> bool:
    """
    Check if the GPU supports cuDNN FlashAttention (requires SM >= 8.0).

    SM 8.0+ = Ampere (A100, A10G, etc.)
    SM 8.9  = Ada Lovelace (L4, L40S, RTX 4090)
    SM 9.0  = Hopper (H100, H200)

    SM 7.5  = Turing (T4) — NOT supported
    SM 7.0  = Volta (V100) — NOT supported

    Cached after first call — zero overhead in the hot path.
    """
    try:
        devices = jax.local_devices()
        for d in devices:
            if d.platform == "gpu":
                # compute_capability returns e.g. "7.5", "8.0", "9.0"
                cc = getattr(d, "compute_capability", None)
                if cc is not None:
                    major = int(str(cc).split(".")[0])
                    return major >= 8
                # If compute_capability is not available, be conservative
                return False
    except Exception:
        pass
    return False


# ------------------------------------------------------------
# Splash Attention (TPU — O(T) memory via Pallas kernel)
# ------------------------------------------------------------

def _splash_causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    TPU Splash Attention with causal masking.

    Uses the Pallas splash attention kernel which computes attention
    in tiles on the TPU's VMEM — O(T) HBM memory instead of O(T²).
    Supports native GQA (K < N, N % K == 0).

    Parameters
    ----------
    q : (B, T, N, H)  — BTNH layout
    k : (B, T, K, H)
    v : (B, T, K, H)

    Returns
    -------
    out : (B, T, N, H)

    Notes
    -----
    The kernel expects BNTH layout internally, so we transpose
    before/after. This is 2 transposes total (vs 6 in the old code),
    and both are pure metadata ops when the tensor is contiguous.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    B, T, N, H = q.shape

    # BTNH → BNTH (kernel expects heads-first, then sequence)
    q = jnp.transpose(q, (0, 2, 1, 3))    # (B, N, T, H)
    k = jnp.transpose(k, (0, 2, 1, 3))    # (B, K, T, H)
    v = jnp.transpose(v, (0, 2, 1, 3))    # (B, K, T, H)

    # Lazy causal mask — computed inside the kernel, zero HBM cost
    causal_mask = splash_attention_mask.CausalMask(shape=(T, T))
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(causal_mask,) * N
    )

    # Block sizes tuned for TPU v5e (128 is the minimum tile size)
    # For seq_len < 512, use smaller blocks to avoid wasting compute
    block = min(512, max(128, T))
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        block_q_dq=block,
        block_kv_dq=block,
    )

    # Build the splash kernel
    # make_splash_mha handles both MHA (K==N) and GQA (K<N) natively
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
    )

    # vmap over batch dimension — kernel processes (N, T, H) slices
    out = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))(
        q, k, v, None  # None = no segment IDs
    )

    # BNTH → BTNH
    return jnp.transpose(out, (0, 2, 1, 3))


# ------------------------------------------------------------
# Attention dispatch — auto-selects kernel by hardware
# ------------------------------------------------------------

def causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    implementation: Optional[str] = None,
) -> jnp.ndarray:
    """
    Causal (autoregressive) attention with hardware-aware kernel dispatch.

    Parameters
    ----------
    q : (B, T, N, H)  — query, N = num_query_heads
    k : (B, S, K, H)  — key,   K = num_kv_heads (K <= N, N % K == 0)
    v : (B, S, K, H)  — value
    implementation : 'splash', 'cudnn', 'xla', or None (auto).
        'splash' → TPU Splash Attention (Pallas kernel, O(T) memory)
        'cudnn'  → cuDNN FlashAttention (GPU Ampere+, O(T) memory)
        'xla'    → XLA einsum with causal mask (O(T²) memory, any backend)
        None     → auto-detect:
                    TPU            → splash
                    GPU (SM >= 8)  → cudnn
                    GPU (SM < 8)   → xla  (T4, V100)
                    CPU            → xla

    Returns
    -------
    out : (B, T, N, H)

    Notes
    -----
    - On TPU, jax.nn.dot_product_attention always uses XLA which
      materializes a full (B,N,T,T) matrix → OOM for real configs.
      Splash Attention avoids this completely.
    - On GPU Ampere+, cuDNN FlashAttention provides O(T) memory.
    - On GPU pre-Ampere (T4, V100), cuDNN flash is not supported.
      Falls back to XLA which materializes the mask but works correctly.
    - scale is NOT applied here — it's applied to Q before this call.
    """
    # Auto-detect hardware
    if implementation is None:
        backend = jax.default_backend()
        if backend == "tpu":
            implementation = "splash"
        elif backend == "gpu":
            # cuDNN FlashAttention requires Ampere+ (SM >= 8.0)
            # T4 (SM 7.5), V100 (SM 7.0) must use XLA fallback
            implementation = "cudnn" if _gpu_supports_cudnn_flash() else "xla"
        else:
            implementation = "xla"

    if implementation == "splash":
        return _splash_causal_attention(q, k, v)

    if implementation == "cudnn":
        return jax.nn.dot_product_attention(
            q, k, v,
            is_causal=True,
            implementation="cudnn",
        )

    # XLA fallback (CPU, or pre-Ampere GPU)
    # implementation=None lets JAX use its default XLA path
    return jax.nn.dot_product_attention(
        q, k, v,
        is_causal=True,
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
    QKV projection dimensions. The attention kernel handles the head
    broadcasting internally for both Splash and cuDNN paths.
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
        # Causal attention (auto hardware dispatch)
        #
        # TPU           → Splash Attention (O(T) memory, Pallas)
        # GPU (Ampere+) → cuDNN FlashAttention (O(T) memory)
        # GPU (T4/V100) → XLA with causal mask (O(T²) memory)
        # CPU           → XLA fallback (O(T²), testing only)
        #
        # GQA is handled natively by all backends.
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
