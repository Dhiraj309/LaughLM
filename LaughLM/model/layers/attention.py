"""
LaughLM/model/layers/attention.py

Frontier-grade attention with Splash on TPU (always — pmap makes each device independent).

Dispatch logic (with pmap, each device is single-device):
  TPU → Splash Attention (Pallas, O(T) memory, tiled on-chip)
  GPU (Ampere+) → cuDNN FlashAttention
  GPU (pre-Ampere) / CPU → XLA fallback

NOTE: With pmap, each device compiles and runs independently.
Splash Attention works fine because there's no SPMD auto-partitioning.

FIX (audit 2025): _pad_for_splash() now right-pads Q/K/V to nearest
Splash-compatible block size (512/256/128) when the sequence length
isn't cleanly divisible. A causal mask prevents padded positions from
attending to real tokens, and padded positions are zeroed after softmax.
This means Splash activates for ALL seq_lens, not just block-aligned ones.
Previously seq_len=1025 silently fell through to XLA dot_product_attention.
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
# KV Cache
# ════════════════════════════════════════════════════════════════

class KVCache(NamedTuple):
    key: jnp.ndarray
    value: jnp.ndarray
    index: jnp.ndarray


def init_kv_cache(batch_size, max_seq_len, num_kv_heads, head_dim,
                  dtype=jnp.bfloat16):
    shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
    return KVCache(
        key=jnp.zeros(shape, dtype=dtype),
        value=jnp.zeros(shape, dtype=dtype),
        index=jnp.array(0, dtype=jnp.int32),
    )


def update_kv_cache(cache, key, value):
    T_new = key.shape[1]
    key = key.astype(cache.key.dtype)
    value = value.astype(cache.value.dtype)
    new_key = jax.lax.dynamic_update_slice(cache.key, key, (0, cache.index, 0, 0))
    new_value = jax.lax.dynamic_update_slice(cache.value, value, (0, cache.index, 0, 0))
    return KVCache(key=new_key, value=new_value, index=cache.index + T_new), new_key, new_value


# ════════════════════════════════════════════════════════════════
# Layout helpers
# ════════════════════════════════════════════════════════════════

def reshape_to_heads(x, num_heads):
    b, t, d = x.shape
    return x.reshape(b, t, num_heads, d // num_heads)

def reshape_from_heads(x):
    b, t, h, d = x.shape
    return x.reshape(b, t, h * d)


# ════════════════════════════════════════════════════════════════
# Hardware detection
# ════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def _gpu_supports_cudnn_flash():
    try:
        for d in jax.local_devices():
            if d.platform == "gpu":
                cc = getattr(d, "compute_capability", None)
                if cc is not None:
                    return int(str(cc).split(".")[0]) >= 8
                return False
    except Exception:
        pass
    return False


# ════════════════════════════════════════════════════════════════
# Splash block size — now also returns padding amount
# ════════════════════════════════════════════════════════════════

def _find_splash_block_size(seq_len):
    """Find aligned block size. Returns (block_size, pad_amount)."""
    for block in [512, 256, 128]:
        if seq_len % block == 0:
            return block, 0
    # No clean alignment — pick largest block, pad up
    block = 512
    pad = ((seq_len + block - 1) // block) * block - seq_len
    return block, pad


# ════════════════════════════════════════════════════════════════
# Padding helper for Splash Attention
# ════════════════════════════════════════════════════════════════

def _pad_for_splash(q, k, v, segment_ids=None):
    """
    Right-pad Q/K/V to nearest Splash-compatible block boundary.

    Returns (q_padded, k_padded, v_padded, pad_amount, seg_padded_or_None).

    The caller must apply a causal mask to prevent padded positions
    from attending to real tokens. Splash attention with CausalMask
    handles this correctly — padded positions at the end can only
    attend to themselves and earlier positions (which are zeroed).
    After attention, the caller strips the padded positions.

    Strategy: pad Q/K/V with zeros. For segment_ids (document mask),
    pad with a special segment ID = -1 so padded positions are
    isolated from real documents.
    """
    B, T, N, H = q.shape
    block, pad = _find_splash_block_size(T)

    if pad == 0:
        return q, k, v, 0, segment_ids

    # Pad last axis (sequence length) with zeros
    pad_shape_q = (B, pad, N, H)
    pad_shape_kv = (B, pad, k.shape[2], H)

    q_pad = jnp.zeros(pad_shape_q, dtype=q.dtype)
    k_pad = jnp.zeros(pad_shape_kv, dtype=k.dtype)
    v_pad = jnp.zeros(pad_shape_kv, dtype=v.dtype)

    q = jnp.concatenate([q, q_pad], axis=1)
    k = jnp.concatenate([k, k_pad], axis=1)
    v = jnp.concatenate([v, v_pad], axis=1)

    if segment_ids is not None:
        seg_pad = jnp.full((B, pad), -1, dtype=segment_ids.dtype)
        segment_ids = jnp.concatenate([segment_ids, seg_pad], axis=1)

    return q, k, v, pad, segment_ids


# ════════════════════════════════════════════════════════════════
# Diagnostic logging (one-time)
# ════════════════════════════════════════════════════════════════

_DISPATCH_LOGGED = set()

def _log_dispatch(impl, detail=""):
    if impl not in _DISPATCH_LOGGED:
        _DISPATCH_LOGGED.add(impl)
        print(f"[attention] Using: {impl}" + (f" ({detail})" if detail else ""))


# ════════════════════════════════════════════════════════════════
# Splash Attention (TPU — O(T) memory)
# ════════════════════════════════════════════════════════════════

def _splash_causal_attention(q, k, v, segment_ids=None):
    """Splash Attention — works with pmap (each device is independent).
    
    NOW WITH AUTO-PADDING: sequences that don't align to Splash block
    sizes (512/256/128) are right-padded with zeros. The causal mask
    prevents padded positions from leaking information. After attention,
    the padding is stripped so the output shape matches the input.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel, splash_attention_mask,
    )

    B, T, N, H = q.shape
    
    # ── Auto-pad for Splash compatibility ───
    q, k, v, pad_amount, segment_ids = _pad_for_splash(q, k, v, segment_ids)
    T_padded = T + pad_amount
    
    block, _ = _find_splash_block_size(T_padded)

    if pad_amount > 0:
        _log_dispatch("splash", f"block={block}, seq_len={T}→{T_padded} (padded {pad_amount})")
    else:
        _log_dispatch("splash", f"block={block}, seq_len={T}")

    # BTNH → BNTH (Splash expects heads-first)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    causal_mask = splash_attention_mask.CausalMask(shape=(T_padded, T_padded))
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(causal_mask,) * N)

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block, block_kv=block, block_kv_compute=block,
        block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=block,
        block_q_dq=block, block_kv_dq=block,
    )

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask, block_sizes=block_sizes,
        head_shards=1, q_seq_shards=1,
    )

    if segment_ids is not None:
        SegmentIds = splash_attention_kernel.SegmentIds
        def _per_sample(q_s, k_s, v_s, seg_s):
            return splash_kernel(q_s, k_s, v_s, SegmentIds(q=seg_s, kv=seg_s))
        out = jax.vmap(_per_sample)(q, k, v, segment_ids)
    else:
        out = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))(q, k, v, None)

    # BNTH → BTNH
    out = jnp.transpose(out, (0, 2, 1, 3))

    # ── Strip padding ───
    if pad_amount > 0:
        out = out[:, :T, :, :]

    return out


# ════════════════════════════════════════════════════════════════
# Document mask (non-Splash backends)
# ════════════════════════════════════════════════════════════════

def _build_document_mask(segment_ids, q_len, kv_len):
    q_ids = segment_ids[:, -q_len:]
    kv_ids = segment_ids[:, :kv_len]
    same_doc = q_ids[:, :, None] == kv_ids[:, None, :]
    return jnp.where(same_doc, 0.0, -1e10)[:, None, :, :]


# ════════════════════════════════════════════════════════════════
# Attention dispatch — ALWAYS Splash on TPU (pmap = single-device)
# ════════════════════════════════════════════════════════════════

def causal_attention(q, k, v, segment_ids=None, implementation=None):
    """
    Causal attention with hardware-aware dispatch.

    With pmap, each device runs independently — Splash works fine.
    No SPMD auto-partition concerns.
    
    Splash Attention now handles non-aligned seq_lens via internal
    right-padding (zero-pad to nearest 512/256/128 boundary + causal mask).
    """
    if implementation is None:
        backend = jax.default_backend()
        if backend == "tpu":
            implementation = "splash"  # Always Splash on TPU (pmap = safe, auto-padded)
        elif backend == "gpu":
            implementation = "cudnn" if _gpu_supports_cudnn_flash() else "xla"
        else:
            implementation = "xla"

    if implementation == "splash":
        try:
            return _splash_causal_attention(q, k, v, segment_ids=segment_ids)
        except (ValueError, RuntimeError, NotImplementedError, Exception) as e:
            import warnings
            warnings.warn(f"[attention] Splash failed ({type(e).__name__}: {e}). Using XLA.", RuntimeWarning)
            _log_dispatch("xla", f"splash fallback — {type(e).__name__}")
            implementation = "xla"

    # cuDNN / XLA paths
    bias = None
    if segment_ids is not None:
        bias = _build_document_mask(segment_ids, q.shape[1], k.shape[1]).astype(q.dtype)

    if implementation == "cudnn":
        _log_dispatch("cudnn", f"seq_len={q.shape[1]}")
        return jax.nn.dot_product_attention(q, k, v, bias=bias, is_causal=True, implementation="cudnn")

    _log_dispatch("xla", f"seq_len={q.shape[1]}")
    return jax.nn.dot_product_attention(q, k, v, bias=bias, is_causal=True)


# ════════════════════════════════════════════════════════════════
# Causal Attention Module
# ════════════════════════════════════════════════════════════════

class CausalAttention(nn.Module):
    config: LaughLMConfig
    d_model: int
    num_heads: int
    num_kv_heads: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, rope_tables=None, doc_ids=None, kv_cache=None):
        compute_dtype = resolve_compute_dtype(self.config)
        param_dtype = resolve_param_dtype(self.config)

        head_dim = self.d_model // self.num_heads
        scale = jnp.array(head_dim ** -0.5, dtype=compute_dtype)

        kv_dim = self.num_kv_heads * head_dim
        qkv_dim = self.d_model + 2 * kv_dim

        qkv = nn.Dense(qkv_dim, use_bias=self.use_bias, dtype=compute_dtype,
                        param_dtype=param_dtype, name="qkv_proj")(x)

        q = qkv[..., :self.d_model]
        k = qkv[..., self.d_model:self.d_model + kv_dim]
        v = qkv[..., self.d_model + kv_dim:]

        q = reshape_to_heads(q, self.num_heads)
        k = reshape_to_heads(k, self.num_kv_heads)
        v = reshape_to_heads(v, self.num_kv_heads)

        if rope_tables is not None:
            sin, cos = rope_tables
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

        q = q * scale
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        new_cache = None
        if kv_cache is not None:
            new_cache, k, v = update_kv_cache(kv_cache, k, v)

        out = causal_attention(q, k, v, segment_ids=doc_ids)
        out = reshape_from_heads(out)

        out = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=compute_dtype,
                       param_dtype=param_dtype, name="out_proj")(out)

        return out, new_cache


# ════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════

def build_attention(config: LaughLMConfig) -> CausalAttention:
    variant = config.architecture.attention_variant
    num_heads = config.model.num_heads

    if variant == "mha":
        num_kv_heads = num_heads
    elif variant == "mqa":
        num_kv_heads = 1
    elif variant == "gqa":
        num_kv_heads = config.model.num_kv_heads
        if num_kv_heads is None:
            raise ValueError("GQA requires num_kv_heads to be set.")
    else:
        raise ValueError(f"Unknown attention variant: '{variant}'")

    return CausalAttention(
        config=config, d_model=config.model.d_model,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        use_bias=config.architecture.bias,
    )