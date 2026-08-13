"""
LaughLM/utils/fused_ops.py

Tokamax Kernel Fusion Dispatcher & Native JAX Fallbacks.

Supported Fused Ops:
1. Fused Cross-Entropy Loss: tokamax.linear_softmax_cross_entropy_loss
2. Fused SwiGLU Activations: tokamax.gated_linear_unit
3. Splash / Ring Attention: tokamax.ops.attention (Pallas Mosaic backend)

Sidecar / Dispatcher pattern ensures zero breaking changes when
kernel_backend == "native" or when tokamax is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Safe Tokamax Import
# ------------------------------------------------------------

try:
    import tokamax
    _TOKAMAX_AVAILABLE = True
except ImportError:
    tokamax = None
    _TOKAMAX_AVAILABLE = False


_TOKAMAX_SWIGLU_FAILED = False
_TOKAMAX_CROSS_ENTROPY_FAILED = False
_TOKAMAX_ATTENTION_FAILED = False


def is_tokamax_available() -> bool:
    """Return whether Tokamax is installed and importable."""
    return _TOKAMAX_AVAILABLE


# ------------------------------------------------------------
# Fused Cross-Entropy Loss
# ------------------------------------------------------------

def fused_linear_softmax_cross_entropy(
    hidden_states: jnp.ndarray,
    targets: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    z_loss: float = 0.0,
    ignore_index: int = -100,
    kernel_backend: str = "native",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fused Linear + Softmax + Cross Entropy loss dispatcher.

    Fuses linear projection, softmax, and loss reduction into TPU SRAM when
    kernel_backend == "tokamax" to prevent materializing massive [B, T, V]
    logit tensors in HBM.

    Returns:
        (loss, z_loss_value)
    """
    global _TOKAMAX_CROSS_ENTROPY_FAILED
    if kernel_backend == "tokamax" and not _TOKAMAX_CROSS_ENTROPY_FAILED:
        if not _TOKAMAX_AVAILABLE:
            logger.warning(
                "[fused_ops] kernel_backend='tokamax' requested, but Tokamax is not installed. "
                "Falling back gracefully to native JAX implementation."
            )
        else:
            try:
                if hasattr(tokamax, "linear_softmax_cross_entropy_loss"):
                    loss, z_loss_val = tokamax.linear_softmax_cross_entropy_loss(
                        hidden_states,
                        targets,
                        weight,
                        bias=bias,
                        z_loss=z_loss,
                        ignore_index=ignore_index,
                    )
                    return loss, z_loss_val
                elif hasattr(tokamax, "linear_cross_entropy"):
                    loss = tokamax.linear_cross_entropy(
                        hidden_states, targets, weight, bias=bias
                    )
                    return loss, jnp.array(0.0, dtype=loss.dtype)
            except Exception as e:
                _TOKAMAX_CROSS_ENTROPY_FAILED = True
                logger.warning(
                    f"[fused_ops] Tokamax fused cross-entropy execution failed ({e}). "
                    "Falling back to native implementation. (Will not attempt again.)"
                )

    # ------------------------------------------------------------
    # Native JAX Fallback
    # ------------------------------------------------------------
    logits = jnp.matmul(hidden_states, weight.T)
    if bias is not None:
        logits = logits + bias

    logits = logits.astype(jnp.float32)
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    log_z = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    log_softmax = shifted_logits - log_z

    valid_mask = (targets != ignore_index).astype(jnp.float32)
    safe_targets = jnp.where(targets == ignore_index, 0, targets)
    
    target_log_probs = jnp.take_along_axis(
        log_softmax, safe_targets[..., None], axis=-1
    ).squeeze(-1)

    loss = -jnp.sum(target_log_probs * valid_mask) / jnp.maximum(
        jnp.sum(valid_mask), 1.0
    )

    z_loss_val = jnp.array(0.0, dtype=logits.dtype)
    if z_loss > 0.0:
        z_loss_val = z_loss * jnp.mean(jnp.square(log_z.squeeze(-1)))
        loss = loss + z_loss_val

    return loss, z_loss_val


# ------------------------------------------------------------
# Fused SwiGLU Activations
# ------------------------------------------------------------

def fused_swiglu(
    gate: jnp.ndarray,
    up: jnp.ndarray,
    kernel_backend: str = "native",
) -> jnp.ndarray:
    """
    Fused SwiGLU activation dispatcher.

    Uses Tokamax on supported GPU backends and native XLA fusion on TPU.
    """
    global _TOKAMAX_SWIGLU_FAILED
    # Tokamax gated_linear_unit is currently GPU-only; use XLA fusion on TPU.
    # Avoid tracing an unsupported custom kernel before falling back.
    if kernel_backend == "tokamax" and jax.default_backend() != "gpu":
        return jax.nn.silu(gate) * up

    if kernel_backend == "tokamax" and not _TOKAMAX_SWIGLU_FAILED:
        if not _TOKAMAX_AVAILABLE:
            logger.warning(
                "[fused_ops] kernel_backend='tokamax' requested, but Tokamax is not installed. "
                "Falling back gracefully to native JAX SwiGLU."
            )
        else:
            try:
                if hasattr(tokamax, "gated_linear_unit"):
                    return tokamax.gated_linear_unit(gate, up, activation="silu")
                elif hasattr(tokamax, "swiglu"):
                    return tokamax.swiglu(gate, up)
            except Exception as e:
                _TOKAMAX_SWIGLU_FAILED = True
                logger.warning(
                    f"[fused_ops] Tokamax fused SwiGLU execution failed ({e}). "
                    "Falling back to native SwiGLU. (Will not attempt again.)"
                )

    # ------------------------------------------------------------
    # Native JAX Fallback
    # ------------------------------------------------------------
    return jax.nn.silu(gate) * up


# ------------------------------------------------------------
# Splash / Ring Attention
# ------------------------------------------------------------

def fused_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
    kernel_backend: str = "native",
) -> jnp.ndarray:
    """
    Fused / Ring / Splash Attention dispatcher.

    Uses tokamax.ops.attention (Pallas Mosaic backend) when kernel_backend == "tokamax"
    for long sequences (T >= 8192).
    """
    global _TOKAMAX_ATTENTION_FAILED
    if kernel_backend == "tokamax" and not _TOKAMAX_ATTENTION_FAILED:
        if not _TOKAMAX_AVAILABLE:
            logger.warning(
                "[fused_ops] kernel_backend='tokamax' requested, but Tokamax is not installed. "
                "Falling back gracefully to native dot_product_attention."
            )
        else:
            try:
                if hasattr(tokamax, "ops") and hasattr(tokamax.ops, "attention"):
                    return tokamax.ops.attention(
                        query, key, value, mask=mask, bias=bias, scale=scale
                    )
                elif hasattr(tokamax, "attention"):
                    return tokamax.attention(
                        query, key, value, mask=mask, bias=bias, scale=scale
                    )
            except Exception as e:
                _TOKAMAX_ATTENTION_FAILED = True
                logger.warning(
                    f"[fused_ops] Tokamax fused attention execution failed ({e}). "
                    "Falling back to native dot_product_attention. (Will not attempt again.)"
                )

    # ------------------------------------------------------------
    # Native JAX Fallback
    # ------------------------------------------------------------
    if scale is None:
        scale = 1.0 / jnp.sqrt(query.shape[-1])

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=mask if mask is not None else bias,
        scale=scale,
    )
