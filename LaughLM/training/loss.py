"""
LaughLM/training/loss.py

Frontier-grade language modeling loss.

Frontier-grade additions
────────────────────────────────────────────
1. GSPMD-safe logical constraints
2. FP32 softmax / CE path
3. Stable masked reductions
4. No hidden all-gathers
5. Vocab-parallel-ready structure
6. Per-token logical loss constraints
7. Stable z-loss accumulation

Design goals
------------
- TPU-stable bf16 training
- fused cross entropy + z-loss
- custom VJP for numerical stability
- packed-sequence compatible
- no one-hot materialization outside CE
- MaxText-style semantics

References
----------
- PaLM (Chowdhery et al.)
- MaxText
- T5X
"""

from typing import Optional

import jax
import jax.numpy as jnp

from LaughLM.distributed.sharding import (
    constrain_logits,
    constrain_loss_tensor,
)


# ------------------------------------------------------------
# Token shifting
# ------------------------------------------------------------

def shift_tokens(
    input_ids: jnp.ndarray,
):
    """
    Shift tokens for causal LM.

    Example
    -------
    [1, 2, 3, 4]

    inputs:
        [1, 2, 3]

    targets:
        [2, 3, 4]
    """

    return (
        input_ids[:, :-1],
        input_ids[:, 1:],
    )


# ------------------------------------------------------------
# Stable fused CE + z-loss
# ------------------------------------------------------------

@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
):
    """
    Numerically stable CE with custom backward.

    Parameters
    ----------
    logits:
        [B, T, V]

    targets:
        one-hot targets:
        [B, T, V]

    Returns
    -------
    loss:
        [B, T]

    z_loss:
        [B, T]
    """

    logits = logits.astype(
        jnp.float32
    )

    logits_sum = (
        jax.scipy.special.logsumexp(
            logits,
            axis=-1,
            keepdims=True,
        )
    )

    log_softmax = (
        logits - logits_sum
    )

    loss = -jnp.sum(
        targets * log_softmax,
        axis=-1,
    )

    log_z = jnp.squeeze(
        logits_sum,
        axis=-1,
    )

    total_z_loss = (
        z_loss
        * jax.lax.square(log_z)
    )

    loss += total_z_loss

    # --------------------------------------------------------
    # Logical constraints
    # --------------------------------------------------------

    loss = constrain_loss_tensor(
        loss
    )

    total_z_loss = (
        constrain_loss_tensor(
            total_z_loss
        )
    )

    return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits,
    targets,
    z_loss=0.0,
):

    logits = logits.astype(
        jnp.float32
    )

    max_logit = logits.max(
        axis=-1,
        keepdims=True,
    )

    shifted = (
        logits - max_logit
    )

    exp_shifted = jnp.exp(
        shifted
    )

    sum_exp = jnp.sum(
        exp_shifted,
        axis=-1,
        keepdims=True,
    )

    log_softmax = (
        shifted
        - jnp.log(sum_exp)
    )

    loss = -jnp.sum(
        targets * log_softmax,
        axis=-1,
    )

    log_z = jnp.squeeze(
        jnp.log(sum_exp)
        + max_logit,
        axis=-1,
    )

    total_z_loss = (
        z_loss
        * jax.lax.square(log_z)
    )

    loss += total_z_loss

    # --------------------------------------------------------
    # Logical constraints
    # --------------------------------------------------------

    loss = constrain_loss_tensor(
        loss
    )

    total_z_loss = (
        constrain_loss_tensor(
            total_z_loss
        )
    )

    return (
        (loss, total_z_loss),
        (
            logits,
            targets,
            z_loss,
            exp_shifted,
            sum_exp,
            log_z,
        ),
    )


def _cross_entropy_with_logits_bwd(
    res,
    g,
):

    g = g[0]

    (
        logits,
        targets,
        z_loss,
        exp_shifted,
        sum_exp,
        log_z,
    ) = res

    deriv = (
        jnp.expand_dims(
            1 + 2 * z_loss * log_z,
            -1,
        )
        * exp_shifted
        / sum_exp
        - targets
    )

    g_logits = (
        jnp.expand_dims(
            g,
            axis=-1,
        )
        * deriv
    )

    return (
        jnp.asarray(
            g_logits,
            logits.dtype,
        ),
        None,
        None,
    )


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd,
    _cross_entropy_with_logits_bwd,
)


# ------------------------------------------------------------
# Main training loss
# ------------------------------------------------------------

def compute_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    z_loss: float = 1e-4,
):
    """
    Frontier-grade LM loss.

    Parameters
    ----------
    logits:
        [B, T, V]

    targets:
        [B, T]

    mask:
        optional [B, T]

    Returns
    -------
    total_loss:
        scalar float32

    metrics:
        dict
    """

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Always compute CE in fp32.
    #
    # bf16 logits are unstable at scale.
    # --------------------------------------------------------

    logits = logits.astype(
        jnp.float32
    )

    # --------------------------------------------------------
    # Logical constraints
    # --------------------------------------------------------

    logits = constrain_logits(
        logits
    )

    vocab_size = logits.shape[-1]

    # --------------------------------------------------------
    # One-hot targets
    #
    # Future:
    # replace with gather-based CE
    # for vocab parallelism.
    # --------------------------------------------------------

    targets_onehot = jax.nn.one_hot(
        targets,
        vocab_size,
        dtype=jnp.float32,
    )

    # --------------------------------------------------------
    # Per-token CE
    # --------------------------------------------------------

    (
        per_token_loss,
        z_loss_value,
    ) = cross_entropy_with_logits(
        logits,
        targets_onehot,
        z_loss=z_loss,
    )

    # --------------------------------------------------------
    # Explicit logical constraints
    # --------------------------------------------------------

    per_token_loss = (
        constrain_loss_tensor(
            per_token_loss
        )
    )

    z_loss_value = (
        constrain_loss_tensor(
            z_loss_value
        )
    )

    # --------------------------------------------------------
    # Masking
    # --------------------------------------------------------

    if mask is not None:

        mask = mask.astype(
            jnp.float32
        )

        mask = constrain_loss_tensor(
            mask
        )

        per_token_loss = (
            per_token_loss * mask
        )

        z_loss_value = (
            z_loss_value * mask
        )

        denom = jnp.maximum(
            jnp.sum(mask),
            1.0,
        )

    else:

        denom = float(
            per_token_loss.size
        )

    # --------------------------------------------------------
    # Stable reductions
    # --------------------------------------------------------

    total_loss = (
        jnp.sum(
            per_token_loss,
            dtype=jnp.float32,
        )
        / denom
    )

    mean_z_loss = (
        jnp.sum(
            z_loss_value,
            dtype=jnp.float32,
        )
        / denom
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "loss": total_loss,
        "z_loss": mean_z_loss,
    }

    return total_loss, metrics
