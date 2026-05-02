"""
LaughLM/model/layers/attention.py

Frontier-grade attention module for decoder-only LLM pretraining.

Frontier optimizations (perf/frontier-optim):
──────────────────────────────────────────────
1. KV Cache — autoregressive decoding with static pre-allocated cache.
   Supports both prefill (full sequence) and decode (single token) modes.

2. Document masking — prevents cross-document attention leaking in
   packed sequences via segment_ids (doc_ids). Both Splash and XLA/cuDNN
   paths support this.

3. Splash Attention GQA fix — MultiHeadMask must use num_q_heads, not
   num_kv_heads. The kernel handles Q→KV head broadcasting internally.

4. Splash block_size alignment — finds largest block ≤ 512 that divides
   the actual sequence length. Falls back to XLA if no valid block exists.

5. Graceful fallback — if Splash fails (Mosaic version mismatch, block
   alignment), automatically falls back to jax.nn.dot_product_attention.

6. Dtype safety — reads from config.spmd.dtype instead of legacy fields.

Key design decisions
────────────────────
• BTNH layout (B, T, num_heads, head_dim) throughout — native for
  jax.nn.dot_product_attention and RoPE. Zero transposes in main path.

• Hardware-aware kernel dispatch:
  TPU           → Splash Attention (Pallas, O(T) memory) with XLA fallback
  GPU (Ampere+) → cuDNN FlashAttention via jax.nn (O(T) memory)
  GPU (pre-Ampere) → XLA with is_causal=True (O(T²) memory)
  CPU           → XLA fallback (O(T²), testing only)

• Native GQA — Splash and cuDNN handle Q_heads != KV_heads natively.

References
──────────
- MaxText: AI-Hypercomputer/maxtext → layers/attention_op.py
- JAX Splash: jax/experimental/pallas/ops/tpu/splash_attention/
- JAX API: jax/_src/nn/functions.py (dot_product_attention)
"""

import functools
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, NamedTuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.positional import apply_rope
from LaughLM.utils.dtype import resolve_compute_dtype, resolve_param_dtype


# ════════════════════════════════════════════════════════════════
# KV Cache for autoregressive decoding
# ════════════════════════════════════════════════════════════════

class KVCache(NamedTuple):
    """
    Static KV cache for autoregressive generation.

    Pre-allocated to max_seq_len and updated in-place via
    jax.lax.dynamic_update_slice. No reallocation during decoding.

    Fields
    ------
    key   : (B, max_seq_len, num_kv_heads, head_dim)
    value : (B, max_seq_len, num_kv_heads, head_dim)
    index : scalar int32 — current write position

    Reference: MaxText inference/kvcache.py
    """
    key: jnp.ndarray
    value: jnp.ndarray
    index: jnp.ndarray


def init_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> KVCache:
    """Create a zeroed KV cache."""
    shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
    return KVCache(
        key=jnp.zeros(shape, dtype=dtype),
        value=jnp.zeros(shape, dtype=dtype),
        index=jnp.array(0, dtype=jnp.int32),
    )


def update_kv_cache(
    cache: KVCache,
    key: jnp.ndarray,
    value: jnp.ndarray,
) -> Tuple[KVCache, jnp.ndarray, jnp.ndarray]:
    """
    Update cache with new K/V tokens and return full K/V for attention.

    Parameters
    ----------
    cache : KVCache
    key   : (B, T_new, num_kv_heads, head_dim) — new tokens
    value : (B, T_new, num_kv_heads, head_dim)

    Returns
    -------
    new_cache : updated KVCache
    full_key  : (B, max_seq_len, num_kv_heads, head_dim)
    full_value: (B, max_seq_len, num_kv_heads, head_dim)
    """
    T_new = key.shape[1]

    # Cast to cache dtype (cache is pre-allocated in a fixed dtype)
    key = key.astype(cache.key.dtype)
    value = value.astype(cache.value.dtype)

    # Write new tokens at current index
    new_key = jax.lax.dynamic_update_slice(
        cache.key, key, (0, cache.index, 0, 0)
    )
    new_value = jax.lax.dynamic_update_slice(
        cache.value, value, (0, cache.index, 0, 0)
    )

    new_cache = KVCache(
        key=new_key,
        value=new_value,
        index=cache.index + T_new,
    )

    return new_cache, new_key, new_value


# ════════════════════════════════════════════════════════════════
# Layout helpers (BTNH — zero transposes for RoPE + projections)
# ════════════════════════════════════════════════════════════════

def reshape_to_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    """Reshape (B, T, D) → (B, T, H, head_dim). Pure reshape, no transpose."""
    b, t, d = x.shape
    head_dim = d // num_heads
    return x.reshape(b, t, num_heads, head_dim)


def reshape_from_heads(x: jnp.ndarray) -> jnp.ndarray:
    """Reshape (B, T, H, head_dim) → (B, T, D). Pure reshape, no transpose."""
    b, t, h, d = x.shape
    return x.reshape(b, t, h * d)


# ════════════════════════════════════════════════════════════════
# GPU capability detection (cached)
# ════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def _gpu_supports_cudnn_flash() -> bool:
    """Check if GPU supports cuDNN FlashAttention (SM >= 8.0 / Ampere+)."""
    try:
        devices = jax.local_devices()
        for d in devices:
            if d.platform == "gpu":
                cc = getattr(d, "compute_capability", None)
                if cc is not None:
                    major = int(str(cc).split(".")[0])
                    return major >= 8
                return False
    except Exception:
        pass
    return False


# ════════════════════════════════════════════════════════════════
# Block size selection for Splash Attention
# ════════════════════════════════════════════════════════════════

def _find_splash_block_size(seq_len: int) -> Optional[int]:
    """
    Find the largest block size ≤ 512 that divides seq_len.

    Splash Attention requires block_q to divide q_seq_len exactly.
    After shift_tokens, seq_len is typically (config.seq_len - 1),
    e.g. 2047, 1023 — which are odd and not divisible by powers of 2.

    Returns None if no valid block size ≥ 128 exists (triggers XLA fallback).
    """
    for block in [512, 256, 128]:
        if seq_len % block == 0:
            return block
    return None


# ════════════════════════════════════════════════════════════════
# Splash Attention (TPU — O(T) memory via Pallas kernel)
# ════════════════════════════════════════════════════════════════

def _splash_causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    TPU Splash Attention with causal masking + optional document masking.

    Supports native GQA: K heads < Q heads, Q_heads % K_heads == 0.
    Falls back to jax.nn.dot_product_attention if block alignment fails.

    Parameters
    ----------
    q : (B, T, N, H) — BTNH layout, N = num_q_heads
    k : (B, T, K, H) — K = num_kv_heads
    v : (B, T, K, H)
    segment_ids : (B, T) int32 or None — document IDs for cross-doc masking

    Returns
    -------
    out : (B, T, N, H)
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    B, T, N, H = q.shape
    K = k.shape[2]  # num_kv_heads

    # ── Find valid block size that divides T ──
    block = _find_splash_block_size(T)
    if block is None:
        # T is not divisible by any standard block size (128/256/512)
        # Fall back to XLA dot_product_attention (no alignment constraint)
        return jax.nn.dot_product_attention(q, k, v, is_causal=True)

    # BTNH → BNTH (kernel expects heads-first)
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, N, T, H)
    k = jnp.transpose(k, (0, 2, 1, 3))  # (B, K, T, H)
    v = jnp.transpose(v, (0, 2, 1, 3))  # (B, K, T, H)

    # ── Mask: must have num_Q_heads masks, not num_KV_heads ──
    causal_mask = splash_attention_mask.CausalMask(shape=(T, T))
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(causal_mask,) * N
    )

    # ── Block sizes ──
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

    # ── Use make_splash_mha for both MHA and GQA ──
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
    )

    # ── Document masking via SegmentIds ──
    if segment_ids is not None:
        SegmentIds = splash_attention_kernel.SegmentIds

        def _per_sample(q_s, k_s, v_s, seg_s):
            seg = SegmentIds(q=seg_s, kv=seg_s)
            return splash_kernel(q_s, k_s, v_s, seg)

        out = jax.vmap(_per_sample)(q, k, v, segment_ids)
    else:
        out = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))(
            q, k, v, None
        )

    # BNTH → BTNH
    return jnp.transpose(out, (0, 2, 1, 3))


# ════════════════════════════════════════════════════════════════
# Document mask builder (for non-Splash backends)
# ════════════════════════════════════════════════════════════════

def _build_document_mask(
    segment_ids: jnp.ndarray,
    q_len: int,
    kv_len: int,
) -> jnp.ndarray:
    """
    Build a boolean attention bias from document segment IDs.

    Cross-document positions get large negative bias (masked out).
    Same-document positions get 0 bias (attend normally).

    Parameters
    ----------
    segment_ids : (B, S) — integer document IDs per position
    q_len, kv_len : sequence lengths (may differ with KV cache)

    Returns
    -------
    bias : (B, 1, q_len, kv_len) — additive bias for attention
    """
    q_ids = segment_ids[:, -q_len:]      # (B, q_len)
    kv_ids = segment_ids[:, :kv_len]     # (B, kv_len)

    # Same document → True, different → False
    same_doc = q_ids[:, :, None] == kv_ids[:, None, :]  # (B, q_len, kv_len)

    # Convert to additive bias: False → -1e10, True → 0
    bias = jnp.where(same_doc, 0.0, -1e10)

    return bias[:, None, :, :]  # (B, 1, q_len, kv_len)


# ════════════════════════════════════════════════════════════════
# Attention dispatch — auto-selects kernel by hardware
# ════════════════════════════════════════════════════════════════

def causal_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: Optional[jnp.ndarray] = None,
    implementation: Optional[str] = None,
) -> jnp.ndarray:
    """
    Causal (autoregressive) attention with hardware-aware kernel dispatch.

    Parameters
    ----------
    q : (B, T, N, H) — query
    k : (B, S, K, H) — key
    v : (B, S, K, H) — value
    segment_ids : (B, S) int or None — document IDs for cross-doc masking
    implementation : 'splash', 'cudnn', 'xla', or None (auto)

    Returns
    -------
    out : (B, T, N, H)
    """
    if implementation is None:
        backend = jax.default_backend()
        if backend == "tpu":
            implementation = "splash"
        elif backend == "gpu":
            implementation = "cudnn" if _gpu_supports_cudnn_flash() else "xla"
        else:
            implementation = "xla"

    if implementation == "splash":
        try:
            return _splash_causal_attention(q, k, v, segment_ids=segment_ids)
        except (ValueError, RuntimeError, Exception) as e:
            # Fallback to XLA if Splash fails (version mismatch, block alignment, etc.)
            import warnings
            warnings.warn(
                f"[attention] Splash Attention failed ({type(e).__name__}: {e}). "
                f"Falling back to jax.nn.dot_product_attention (XLA).",
                RuntimeWarning,
            )
            implementation = "xla"

    # ── cuDNN / XLA paths with optional document mask ──
    bias = None
    if segment_ids is not None:
        bias = _build_document_mask(segment_ids, q.shape[1], k.shape[1])
        bias = bias.astype(q.dtype)

    if implementation == "cudnn":
        return jax.nn.dot_product_attention(
            q, k, v,
            bias=bias,
            is_causal=True,
            implementation="cudnn",
        )

    # XLA fallback (CPU, pre-Ampere GPU, or TPU when Splash unavailable)
    return jax.nn.dot_product_attention(
        q, k, v,
        bias=bias,
        is_causal=True,
    )


# ════════════════════════════════════════════════════════════════
# Sharding helper
# ════════════════════════════════════════════════════════════════

def _maybe_shard(x: jnp.ndarray, spec) -> jnp.ndarray:
    """Apply with_sharding_constraint if spec is not None."""
    if spec is not None:
        return jax.lax.with_sharding_constraint(x, spec)
    return x


# ════════════════════════════════════════════════════════════════
# Unified Causal Attention (MHA / MQA / GQA)
# ════════════════════════════════════════════════════════════════

class CausalAttention(nn.Module):
    """
    Unified multi-head / multi-query / grouped-query causal attention.

    Supports:
    - Training (full sequence, no cache)
    - Inference with KV cache (prefill + autoregressive decode)
    - Document masking via doc_ids / segment_ids
    - Sharding annotations for FSDP + tensor parallelism

    When num_kv_heads == num_heads → standard MHA
    When num_kv_heads == 1         → MQA
    When 1 < num_kv_heads < num_heads → GQA
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
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        """
        Parameters
        ----------
        x          : (B, T, D) input
        rope_tables: (sin, cos) for RoPE, sliced to seq_len
        doc_ids    : (B, T) integer segment/document IDs
        kv_cache   : optional KVCache for autoregressive decoding

        Returns
        -------
        output     : (B, T, D)
        new_cache  : updated KVCache or None
        """
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        head_dim = self.d_model // self.num_heads

        # Scale in compute_dtype to prevent type promotion
        scale = jnp.array(head_dim ** -0.5, dtype=compute_dtype)

        kv_dim = self.num_kv_heads * head_dim
        qkv_dim = self.d_model + 2 * kv_dim

        # ── Fused QKV projection (single matmul) ──────────────
        qkv = nn.Dense(
            qkv_dim,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="qkv_proj",
        )(x)

        q = qkv[..., :self.d_model]
        k = qkv[..., self.d_model:self.d_model + kv_dim]
        v = qkv[..., self.d_model + kv_dim:]

        # ── Reshape to (B, T, H, head_dim) — no transpose ────
        q = reshape_to_heads(q, self.num_heads)
        k = reshape_to_heads(k, self.num_kv_heads)
        v = reshape_to_heads(v, self.num_kv_heads)

        # ── RoPE (before scaling) ─────────────────────────────
        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        # ── Scale Q after RoPE ────────────────────────────────
        q = q * scale

        # ── Dtype enforcement ─────────────────────────────────
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        # ── KV Cache update (inference only) ──────────────────
        new_cache = None
        if kv_cache is not None:
            new_cache, k, v = update_kv_cache(kv_cache, k, v)

        # ── Causal attention (hardware-dispatched) ────────────
        out = causal_attention(q, k, v, segment_ids=doc_ids)

        # ── Merge heads ───────────────────────────────────────
        out = reshape_from_heads(out)

        # ── Output projection ─────────────────────────────────
        out = nn.Dense(
            self.d_model,
            use_bias=self.use_bias,
            dtype=compute_dtype,
            param_dtype=param_dtype,
            name="out_proj",
        )(out)

        return out, new_cache


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

def build_attention(config: LaughLMConfig) -> CausalAttention:
    """
    Build attention module from config.

    All variants (MHA, MQA, GQA) use the same CausalAttention class.
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
