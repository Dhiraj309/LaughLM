import jax
import jax.numpy as jnp
import optax
from typing import Optional, Dict, Tuple


# ------------------------------------------------------------
# Token shifting for causal LM
# ------------------------------------------------------------

def shift_tokens(input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift token IDs for causal language modeling.

    For a sequence [t0, t1, t2, t3]:
        inputs  = [t0, t1, t2]   ← fed into the model
        targets = [t1, t2, t3]   ← ground truth to predict

    FIX (3 bugs):
      1. 'input_idss' → 'input_ids'  (typo)
      2. '[:, -1]' → '[:, :-1]'     (was slicing last token only)
      3. 'target' → 'targets'        (undefined variable)
    """
    inputs  = input_ids[:, :-1]   # [B, T-1]
    targets = input_ids[:, 1:]    # [B, T-1]
    return inputs, targets


# ------------------------------------------------------------
# Cross-entropy loss
# ------------------------------------------------------------

def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Token-level cross-entropy loss, averaged over unmasked positions.

    Parameters
    ----------
    logits  : [B, T, V]  — raw model output (float32)
    targets : [B, T]     — integer token IDs
    mask    : [B, T]     — float mask, 1.0 = include, 0.0 = exclude (padding / cross-doc)
                           When None, all positions are included.

    Returns a scalar.

    FIX (2 bugs):
      1. 'jnp.nn.log_softmax' → uses optax which calls jax.nn internally
         (jnp has no .nn submodule — that was a NameError)
      2. Loss sign: was positive log-likelihood (maximizes wrong answers).
         Now correctly negative cross-entropy (minimizes loss).

    Using optax.softmax_cross_entropy_with_integer_labels:
      - Internally uses logsumexp for numerical stability
      - Avoids materializing one-hot [B, T, V] tensor (expensive for large vocab)
      - Returns per-token losses [B, T], we average over unmasked positions
    """

    # Per-token cross-entropy: [B, T]
    # This correctly computes: -log P(target | context)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,     # [B, T, V]
        labels=targets,    # [B, T]  integers
    )

    if mask is not None:
        # Mask out padding / cross-document positions
        per_token_loss = per_token_loss * mask
        # Average only over included positions
        loss = jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        loss = jnp.mean(per_token_loss)

    return loss


# ------------------------------------------------------------
# Z-loss regularization
# ------------------------------------------------------------

def z_loss(
    logits: jnp.ndarray,
    coeff: float = 1e-4,
) -> jnp.ndarray:
    """
    Z-loss regularization (Chowdhery et al., PaLM).

    Penalizes large logit magnitudes to prevent logit drift.
    log Z = logsumexp(logits) → Z-loss = coeff × E[log²Z]

    This keeps logits in a numerically stable range throughout training
    and is particularly useful for very deep models or long runs.
    The coefficient 1e-4 is the standard value from PaLM.

    FIX: was called as z_loss(loss, z_loss_coeff) — 'loss' was undefined.
    Signature now explicitly takes logits as first argument.
    """
    # logsumexp over vocab dimension
    log_z = jax.nn.logsumexp(logits, axis=-1)   # [B, T]
    return coeff * jnp.mean(log_z ** 2)


# ------------------------------------------------------------
# Combined training loss
# ------------------------------------------------------------

def compute_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    zloss_coeff: float = 1e-4,
):

    ce = cross_entropy_loss(logits, targets, mask)
    zl = z_loss(logits, zloss_coeff)
    total = ce + zl

    metrics = {
        "cross_entropy": ce,
        "z_loss": zl,
        "total": total,
    }

    return total, metrics
