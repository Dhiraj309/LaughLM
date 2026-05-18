"""
LaughLM/training/loss.py
"""

from typing import Optional

import jax
import jax.numpy as jnp

from LaughLM.distributed.sharding import (
    constrain_logits,
    constrain_loss_tensor,
)


# ─────────────────────────────────────────────────────────────
# Token shifting
# ─────────────────────────────────────────────────────────────

def shift_tokens(
    input_ids: jnp.ndarray,
):

    return (
        input_ids[:, :-1],
        input_ids[:, 1:],
    )


# ─────────────────────────────────────────────────────────────
# Stable CE
# ─────────────────────────────────────────────────────────────

@jax.custom_vjp
def cross_entropy_with_logits(
    logits,
    targets,
    z_loss=0.0,
):

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

    log_softmax = logits - logits_sum

    loss = -jnp.sum(
        targets * log_softmax,
        axis=-1,
    )

    log_z = jnp.squeeze(
        logits_sum,
        axis=-1,
    )

    z_loss_value = (
        z_loss
        * jax.lax.square(log_z)
    )

    loss += z_loss_value

    return loss, z_loss_value


def _cross_entropy_with_logits_fwd(
    logits,
    targets,
    z_loss=0.0,
):

    logits = logits.astype(
        jnp.float32
    )

    max_logit = jnp.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    shifted = logits - max_logit

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

    z_loss_value = (
        z_loss
        * jax.lax.square(log_z)
    )

    loss += z_loss_value

    return (
        (loss, z_loss_value),
        (
            targets,
            exp_shifted,
            sum_exp,
            log_z,
            z_loss,
            logits.dtype,
        ),
    )


def _cross_entropy_with_logits_bwd(
    res,
    g,
):

    g = g[0]

    (
        targets,
        exp_shifted,
        sum_exp,
        log_z,
        z_loss,
        logits_dtype,
    ) = res

    softmax = (
        exp_shifted / sum_exp
    )

    deriv = (
        (
            1
            + 2 * z_loss * log_z
        )[..., None]
        * softmax
        - targets
    )

    g_logits = (
        g[..., None]
        * deriv
    )

    return (
        g_logits.astype(
            logits_dtype
        ),
        None,
        None,
    )


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd,
    _cross_entropy_with_logits_bwd,
)


# ─────────────────────────────────────────────────────────────
# Main loss
# ─────────────────────────────────────────────────────────────

def compute_loss(
    logits,
    targets,
    mask: Optional[jnp.ndarray] = None,
    z_loss: float = 1e-4,
):

    logits = logits.astype(
        jnp.float32
    )

    logits = constrain_logits(
        logits
    )

    vocab_size = logits.shape[-1]

    targets_onehot = jax.nn.one_hot(
        targets,
        vocab_size,
        dtype=jnp.float32,
    )

    (
        per_token_loss,
        z_loss_value,
    ) = cross_entropy_with_logits(
        logits,
        targets_onehot,
        z_loss=z_loss,
    )

    per_token_loss = (
        constrain_loss_tensor(
            per_token_loss
        )
    )

    if mask is not None:

        mask = mask.astype(
            jnp.float32
        )

        mask = constrain_loss_tensor(
            mask
        )

        per_token_loss *= mask

        z_loss_value *= mask

        denom = jnp.maximum(
            jnp.sum(mask),
            1.0,
        )

    else:

        denom = jnp.asarray(
            per_token_loss.size,
            dtype=jnp.float32,
        )

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

    metrics = {
        "loss": total_loss,
        "z_loss": mean_z_loss,
    }

    return total_loss, metrics
