"""
LaughLM/training/loss.py

Loss functions for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Clean API — compute_loss() is the single entry point.
   Returns (total_loss, metrics_dict).

2. Z-loss regularization — penalizes large logit magnitudes
   for training stability (Chowdhery et al., PaLM 2022).

3. Softmax cross-entropy via optax — uses logsumexp internally
   for numerical stability, avoids materializing one-hot [B,T,V].

References:
  Z-loss: Chowdhery et al. "PaLM" (2022)
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Dict, Tuple


# ────────────────────────────────────────────────────────────────
# Token shifting for causal LM
# ────────────────────────────────────────────────────────────────

def shift_tokens(input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift token IDs for causal language modeling.

    [t0, t1, t2, t3] →
        inputs  = [t0, t1, t2]
        targets = [t1, t2, t3]
    """
    inputs  = input_ids[:, :-1]   # [B, T-1]
    targets = input_ids[:, 1:]    # [B, T-1]
    return inputs, targets


# ────────────────────────────────────────────────────────────────
# Cross-entropy loss
# ────────────────────────────────────────────────────────────────

def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Token-level cross-entropy loss.

    Parameters
    ----------
    logits  : [B, T, V] — raw model output (should be float32)
    targets : [B, T]    — integer token IDs
    mask    : [B, T]    — 1.0 = include, 0.0 = exclude (padding)

    Returns
    -------
    scalar loss
    """
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets,
    )

    if mask is not None:
        per_token_loss = per_token_loss * mask
        loss = jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        loss = jnp.mean(per_token_loss)

    return loss


# ────────────────────────────────────────────────────────────────
# Z-loss regularization
# ────────────────────────────────────────────────────────────────

def z_loss(
    logits: jnp.ndarray,
    coeff: float = 1e-4,
) -> jnp.ndarray:
    """
    Z-loss regularization (Chowdhery et al., PaLM 2022).

    Penalizes large logit magnitudes:
        z_loss = coeff * E[log²(Z)]  where Z = sum(exp(logits))

    Keeps logits in a numerically stable range throughout training.
    """
    log_z = jax.nn.logsumexp(logits, axis=-1)   # [B, T]
    return coeff * jnp.mean(log_z ** 2)


# ────────────────────────────────────────────────────────────────
# Combined training loss
# ────────────────────────────────────────────────────────────────

def compute_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    zloss_coeff: float = 1e-4,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute total training loss = cross_entropy + z_loss.

    Parameters
    ----------
    logits      : [B, T, V] in float32
    targets     : [B, T] integer labels
    mask        : optional [B, T] mask
    zloss_coeff : z-loss coefficient (default 1e-4 from PaLM)

    Returns
    -------
    total_loss : scalar
    metrics    : dict with individual loss components
    """
    ce = cross_entropy_loss(logits, targets, mask)
    zl = z_loss(logits, zloss_coeff)
    total = ce + zl

    metrics = {
        "cross_entropy": ce,
        "z_loss": zl,
        "total": total,
    }

    return total, metrics
