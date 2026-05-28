"""
LaughLM/training/loss.py

Loss functions for LaughLM.

PMAP runtime fix:
- Keep the old dense-logits CE path for compatibility/tests.
- Add hidden-state LM loss path.
- Add exact chunked vocab cross entropy to avoid materializing [B, T, vocab].

Important:
Chunked CE is still exact dense-softmax CE. It does NOT approximate the
softmax denominator. It scans over vocab chunks and accumulates logsumexp.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jax import lax

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
# Stable CE — legacy dense logits path
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
        g_logits,
        None,
        None,
    )


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd,
    _cross_entropy_with_logits_bwd,
)


# ─────────────────────────────────────────────────────────────
# Legacy dense logits loss
# ─────────────────────────────────────────────────────────────

def compute_loss(
    logits,
    targets,
    mask: Optional[jnp.ndarray] = None,
    z_loss: float = 1e-4,
):
    """
    Dense sparse-label CE from already materialized logits.

    Correct but memory-heavy because logits are [B, T, vocab].
    PMAP training should use compute_lm_loss_from_hidden(...).
    """
    logits = logits.astype(jnp.float32)
    logits = constrain_logits(logits)

    log_z = jax.scipy.special.logsumexp(
        logits,
        axis=-1,
    )

    target_logits = jnp.take_along_axis(
        logits,
        targets[..., None],
        axis=-1,
    )[..., 0]

    per_token_xent = (
        log_z
        - target_logits
    )

    z_loss_value = (
        z_loss
        * jax.lax.square(log_z)
    )

    per_token_loss = (
        per_token_xent
        + z_loss_value
    )

    per_token_loss = constrain_loss_tensor(
        per_token_loss
    )

    z_loss_value = constrain_loss_tensor(
        z_loss_value
    )

    if mask is not None:
        mask = constrain_loss_tensor(
            mask.astype(jnp.float32)
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
        jnp.sum(per_token_loss, dtype=jnp.float32)
        / denom
    )

    mean_z_loss = (
        jnp.sum(z_loss_value, dtype=jnp.float32)
        / denom
    )

    return total_loss, {
        "loss": total_loss,
        "z_loss": mean_z_loss,
    }


# ─────────────────────────────────────────────────────────────
# Hidden-state LM-head helpers
# ─────────────────────────────────────────────────────────────

def _ceil_to_multiple(
    x: int,
    multiple: int,
) -> int:
    return (
        (x + multiple - 1)
        // multiple
    ) * multiple


def _pad_axis(
    x: jnp.ndarray,
    padded_size: int,
    axis: int,
) -> jnp.ndarray:
    current = x.shape[axis]
    pad = padded_size - current

    if pad == 0:
        return x

    pad_width = [
        (0, 0)
        for _ in range(x.ndim)
    ]

    pad_width[axis] = (
        0,
        pad,
    )

    return jnp.pad(
        x,
        pad_width,
        mode="constant",
        constant_values=0,
    )


def _infer_lm_head_layout(
    lm_head_kernel: jnp.ndarray,
    hidden_size: int,
) -> tuple[int, bool]:
    """
    Returns:
        vocab_size
        is_vocab_major

    Supported layouts:
      tied embedding: [vocab, hidden]
      dense kernel:   [hidden, vocab]
    """
    if lm_head_kernel.ndim != 2:
        raise ValueError(
            "lm_head_kernel must be rank-2. "
            f"Got shape={lm_head_kernel.shape}"
        )

    if lm_head_kernel.shape[1] == hidden_size:
        return int(lm_head_kernel.shape[0]), True

    if lm_head_kernel.shape[0] == hidden_size:
        return int(lm_head_kernel.shape[1]), False

    raise ValueError(
        "Could not infer lm_head layout. Expected [vocab, hidden] "
        "or [hidden, vocab]. "
        f"Got shape={lm_head_kernel.shape}, hidden_size={hidden_size}"
    )


def _as_vocab_major(
    lm_head_kernel: jnp.ndarray,
    *,
    hidden_size: int,
) -> tuple[jnp.ndarray, int]:
    vocab_size, is_vocab_major = _infer_lm_head_layout(
        lm_head_kernel,
        hidden_size,
    )

    if is_vocab_major:
        return lm_head_kernel, vocab_size

    return jnp.swapaxes(
        lm_head_kernel,
        0,
        1,
    ), vocab_size


def _valid_target_mask(
    targets: jnp.ndarray,
    *,
    vocab_size: int,
    ignore_index: int,
    mask: Optional[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    targets = targets.astype(jnp.int32)

    valid = (
        (targets != ignore_index)
        & (targets >= 0)
        & (targets < vocab_size)
    )

    if mask is not None:
        valid = valid & mask.astype(bool)

    safe_targets = jnp.where(
        valid,
        targets,
        0,
    )

    return valid, safe_targets


# ─────────────────────────────────────────────────────────────
# Dense hidden-state LM loss
# ─────────────────────────────────────────────────────────────

def dense_lm_loss_from_hidden(
    *,
    hidden_states: jnp.ndarray,
    targets: jnp.ndarray,
    lm_head_kernel: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    lm_head_bias: Optional[jnp.ndarray] = None,
    z_loss: float = 1e-4,
    ignore_index: int = -100,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Dense reference path:
      hidden [B, T, D] @ lm_head [V, D].T -> logits [B, T, V]

    Correct but memory-heavy.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [B, T, D], got {hidden_states.shape}"
        )

    B, T, D = hidden_states.shape

    if targets.shape != (B, T):
        raise ValueError(
            f"targets must be [B, T]={B, T}, got {targets.shape}"
        )

    w_vocab_major, vocab_size = _as_vocab_major(
        lm_head_kernel,
        hidden_size=D,
    )

    logits = jnp.einsum(
        "btd,vd->btv",
        hidden_states,
        w_vocab_major,
        precision=lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )

    if lm_head_bias is not None:
        logits = (
            logits
            + lm_head_bias.astype(jnp.float32)
        )

    logits = logits.astype(jnp.float32)
    logits = constrain_logits(logits)

    valid, safe_targets = _valid_target_mask(
        targets,
        vocab_size=vocab_size,
        ignore_index=ignore_index,
        mask=mask,
    )

    log_z = jax.scipy.special.logsumexp(
        logits,
        axis=-1,
    )

    target_logits = jnp.take_along_axis(
        logits,
        safe_targets[..., None],
        axis=-1,
    )[..., 0]

    per_token_xent = (
        log_z
        - target_logits
    )

    z_loss_value = (
        z_loss
        * jax.lax.square(log_z)
    )

    per_token_loss = (
        per_token_xent
        + z_loss_value
    )

    per_token_loss = jnp.where(
        valid,
        per_token_loss,
        0.0,
    )

    z_loss_value = jnp.where(
        valid,
        z_loss_value,
        0.0,
    )

    per_token_loss = constrain_loss_tensor(
        per_token_loss
    )

    z_loss_value = constrain_loss_tensor(
        z_loss_value
    )

    denom = jnp.maximum(
        jnp.sum(valid.astype(jnp.float32)),
        1.0,
    )

    total_loss = (
        jnp.sum(per_token_loss, dtype=jnp.float32)
        / denom
    )

    mean_z_loss = (
        jnp.sum(z_loss_value, dtype=jnp.float32)
        / denom
    )

    bad_label_count = jnp.sum(
        (
            (targets != ignore_index)
            & (
                (targets < 0)
                | (targets >= vocab_size)
            )
        ).astype(jnp.float32)
    )

    return total_loss, {
        "loss": total_loss,
        "z_loss": mean_z_loss,
        "valid_tokens": denom,
        "bad_label_count": bad_label_count,
    }


# ─────────────────────────────────────────────────────────────
# Chunked exact LM loss
# ─────────────────────────────────────────────────────────────

def chunked_lm_loss_from_hidden(
    *,
    hidden_states: jnp.ndarray,
    targets: jnp.ndarray,
    lm_head_kernel: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    lm_head_bias: Optional[jnp.ndarray] = None,
    chunk_size: int = 4096,
    z_loss: float = 1e-4,
    ignore_index: int = -100,
    remat_chunks: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Exact sparse-label LM CE without materializing [B, T, vocab].

    Computes:
      CE = logsumexp(hidden @ W.T) - target_logit

    The logsumexp denominator is still over the full vocabulary.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be > 0, got {chunk_size}"
        )

    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [B, T, D], got {hidden_states.shape}"
        )

    B, T, D = hidden_states.shape

    if targets.shape != (B, T):
        raise ValueError(
            f"targets must be [B, T]={B, T}, got {targets.shape}"
        )

    w_vocab_major, vocab_size = _as_vocab_major(
        lm_head_kernel,
        hidden_size=D,
    )

    if lm_head_bias is not None and lm_head_bias.shape != (vocab_size,):
        raise ValueError(
            f"lm_head_bias must be [{vocab_size}], got {lm_head_bias.shape}"
        )

    padded_vocab_size = _ceil_to_multiple(
        vocab_size,
        chunk_size,
    )

    num_chunks = (
        padded_vocab_size
        // chunk_size
    )

    w_vocab_major = _pad_axis(
        w_vocab_major,
        padded_vocab_size,
        axis=0,
    )

    if lm_head_bias is not None:
        lm_head_bias = _pad_axis(
            lm_head_bias,
            padded_vocab_size,
            axis=0,
        )

    hidden_flat = hidden_states.reshape(
        B * T,
        D,
    )

    targets_flat = targets.astype(jnp.int32).reshape(
        B * T,
    )

    if mask is None:
        mask_flat = jnp.ones(
            (B * T,),
            dtype=bool,
        )
    else:
        if mask.shape != (B, T):
            raise ValueError(
                f"mask must be [B, T]={B, T}, got {mask.shape}"
            )

        mask_flat = mask.astype(bool).reshape(
            B * T,
        )

    label_present = (
        targets_flat != ignore_index
    )

    label_in_range = (
        (targets_flat >= 0)
        & (targets_flat < vocab_size)
    )

    active = (
        mask_flat
        & label_present
        & label_in_range
    )

    safe_targets = jnp.where(
        active,
        targets_flat,
        0,
    )

    neg_inf = jnp.asarray(
        -jnp.inf,
        dtype=jnp.float32,
    )

    init_max = jnp.full(
        (B * T,),
        neg_inf,
        dtype=jnp.float32,
    )

    init_sum = jnp.zeros(
        (B * T,),
        dtype=jnp.float32,
    )

    init_target_logits = jnp.zeros(
        (B * T,),
        dtype=jnp.float32,
    )

    def body_impl(
        carry,
        chunk_idx,
    ):
        running_max, running_sum, target_logits = carry

        start = (
            chunk_idx
            * chunk_size
        )

        w_chunk = lax.dynamic_slice_in_dim(
            w_vocab_major,
            start_index=start,
            slice_size=chunk_size,
            axis=0,
        )

        logits_chunk = jnp.einsum(
            "nd,vd->nv",
            hidden_flat,
            w_chunk,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

        if lm_head_bias is not None:
            bias_chunk = lax.dynamic_slice_in_dim(
                lm_head_bias,
                start_index=start,
                slice_size=chunk_size,
                axis=0,
            )

            logits_chunk = (
                logits_chunk
                + bias_chunk.astype(jnp.float32)[None, :]
            )

        vocab_ids = (
            start
            + jnp.arange(
                chunk_size,
                dtype=jnp.int32,
            )
        )

        valid_vocab = (
            vocab_ids < vocab_size
        )

        logits_chunk = jnp.where(
            valid_vocab[None, :],
            logits_chunk,
            neg_inf,
        )

        chunk_max = jnp.max(
            logits_chunk,
            axis=-1,
        )

        new_max = jnp.maximum(
            running_max,
            chunk_max,
        )

        old_scale = jnp.exp(
            running_max
            - new_max
        )

        chunk_scale = jnp.exp(
            logits_chunk
            - new_max[:, None]
        )

        new_sum = (
            running_sum
            * old_scale
            + jnp.sum(
                chunk_scale,
                axis=-1,
            )
        )

        in_chunk = (
            active
            & (safe_targets >= start)
            & (safe_targets < start + chunk_size)
        )

        local_targets = jnp.clip(
            safe_targets - start,
            0,
            chunk_size - 1,
        )

        chunk_target_logits = jnp.take_along_axis(
            logits_chunk,
            local_targets[:, None],
            axis=-1,
        )[:, 0]

        target_logits = jnp.where(
            in_chunk,
            chunk_target_logits,
            target_logits,
        )

        return (
            new_max,
            new_sum,
            target_logits,
        ), None

    body = (
        jax.checkpoint(body_impl)
        if remat_chunks
        else body_impl
    )

    (
        final_max,
        final_sum,
        target_logits,
    ), _ = lax.scan(
        body,
        (
            init_max,
            init_sum,
            init_target_logits,
        ),
        jnp.arange(
            num_chunks,
            dtype=jnp.int32,
        ),
    )

    log_z = (
        final_max
        + jnp.log(final_sum)
    )

    per_token_xent = (
        log_z
        - target_logits
    )

    z_loss_value = (
        z_loss
        * jax.lax.square(log_z)
    )

    per_token_loss = (
        per_token_xent
        + z_loss_value
    )

    per_token_loss = jnp.where(
        active,
        per_token_loss,
        0.0,
    )

    z_loss_value = jnp.where(
        active,
        z_loss_value,
        0.0,
    )

    per_token_loss = per_token_loss.reshape(
        B,
        T,
    )

    z_loss_value = z_loss_value.reshape(
        B,
        T,
    )

    per_token_loss = constrain_loss_tensor(
        per_token_loss
    )

    z_loss_value = constrain_loss_tensor(
        z_loss_value
    )

    denom = jnp.maximum(
        jnp.sum(active.astype(jnp.float32)),
        1.0,
    )

    total_loss = (
        jnp.sum(per_token_loss, dtype=jnp.float32)
        / denom
    )

    mean_z_loss = (
        jnp.sum(z_loss_value, dtype=jnp.float32)
        / denom
    )

    bad_label_count = jnp.sum(
        (
            label_present
            & (
                (targets_flat < 0)
                | (targets_flat >= vocab_size)
            )
        ).astype(jnp.float32)
    )

    return total_loss, {
        "loss": total_loss,
        "z_loss": mean_z_loss,
        "valid_tokens": denom,
        "bad_label_count": bad_label_count,
        "logits_chunk_size": jnp.asarray(
            chunk_size,
            dtype=jnp.int32,
        ),
        "logits_num_chunks": jnp.asarray(
            num_chunks,
            dtype=jnp.int32,
        ),
    }


# ─────────────────────────────────────────────────────────────
# Unified hidden-state LM loss
# ─────────────────────────────────────────────────────────────

def compute_lm_loss_from_hidden(
    *,
    hidden_states: jnp.ndarray,
    targets: jnp.ndarray,
    lm_head_kernel: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    lm_head_bias: Optional[jnp.ndarray] = None,
    chunked_logits: bool = False,
    logits_chunk_size: int = 4096,
    z_loss: float = 1e-4,
    ignore_index: int = -100,
    remat_logits_chunks: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Main LM loss entrypoint for training.
    """
    if chunked_logits:
        return chunked_lm_loss_from_hidden(
            hidden_states=hidden_states,
            targets=targets,
            lm_head_kernel=lm_head_kernel,
            mask=mask,
            lm_head_bias=lm_head_bias,
            chunk_size=logits_chunk_size,
            z_loss=z_loss,
            ignore_index=ignore_index,
            remat_chunks=remat_logits_chunks,
        )

    return dense_lm_loss_from_hidden(
        hidden_states=hidden_states,
        targets=targets,
        lm_head_kernel=lm_head_kernel,
        mask=mask,
        lm_head_bias=lm_head_bias,
        z_loss=z_loss,
        ignore_index=ignore_index,
    )
