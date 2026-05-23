"""
LaughLM/training/train_step.py

PMAP-native train/eval step for LaughLM.

Design:
- replicated params/state
- local forward/backward per device
- direct fast path when grad_accum == 1
- gradient accumulation via lax.scan when grad_accum > 1
- cross-device grad averaging via pmean
- FP32 grad accumulation
- Optax optimizer updates
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from LaughLM.training.loss import (
    shift_tokens,
    compute_loss,
)

from LaughLM.distributed.sharding import (
    constrain_batch,
    constrain_logits,
)


Params = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ============================================================
# Train step factory
# ============================================================

def create_train_step(
    *,
    model,
    optimizer,
    grad_accum: int,
    max_grad_norm: float = 1.0,
):

    if grad_accum <= 0:
        raise ValueError(
            f"grad_accum must be > 0, got {grad_accum}"
        )

    # --------------------------------------------------------
    # Loss fn
    # --------------------------------------------------------

    def loss_fn(
        params,
        micro_batch,
    ):
        inputs, targets = shift_tokens(
            micro_batch
        )

        logits, _ = model.apply(
            {"params": params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, metrics = compute_loss(
            logits,
            targets,
        )

        return loss, metrics

    # --------------------------------------------------------
    # Shared optimizer update
    # --------------------------------------------------------

    def apply_update(
        state,
        params,
        grads,
        loss,
        batch,
    ):
        # ----------------------------------------------------
        # Cross-device gradient averaging
        # ----------------------------------------------------

        grads = jax.lax.pmean(
            grads,
            axis_name="data",
        )

        # ----------------------------------------------------
        # Cross-device loss averaging
        # ----------------------------------------------------

        loss = jax.lax.pmean(
            loss.astype(jnp.float32),
            axis_name="data",
        )

        # ----------------------------------------------------
        # Global grad norm after pmean
        # ----------------------------------------------------

        grad_norm = optax.global_norm(
            grads
        ).astype(jnp.float32)

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm
            / jnp.maximum(
                grad_norm,
                1e-6,
            ),
        )

        grads = jax.tree_util.tree_map(
            lambda g: g * clip_scale,
            grads,
        )

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        updates, new_opt_state = optimizer.update(
            grads,
            state.opt_state,
            params,
        )

        new_params = optax.apply_updates(
            params,
            updates,
        )

        # ----------------------------------------------------
        # Tokens processed
        #
        # PMAP local batch shape:
        #   grad_accum == 1: [1, micro_batch, seq_len]
        #   grad_accum > 1:  [grad_accum, micro_batch, seq_len]
        #
        # Keep int64 to avoid overflow on 30B-token runs.
        # Host-side accounting in trainer.py remains the source of
        # truth for logging/checkpoint metadata.
        # ----------------------------------------------------

        local_tokens = (
            batch.shape[0]
            * batch.shape[1]
            * batch.shape[2]
        )

        global_tokens = jax.lax.psum(
            jnp.asarray(
                local_tokens,
                dtype=jnp.int64,
            ),
            axis_name="data",
        )

        new_state = state.apply_grad_step(
            params=new_params,
            opt_state=new_opt_state,
            tokens_in_step=global_tokens,
        )

        metrics = {
            "loss": loss.astype(jnp.float32),
            "grad_norm": grad_norm.astype(jnp.float32),
        }

        return new_state, metrics

    # --------------------------------------------------------
    # PMAP train step
    # --------------------------------------------------------

    def train_step(
        state,
        batch,
    ):
        batch = constrain_batch(
            batch
        )

        params = state.params

        # ----------------------------------------------------
        # Fast path: no gradient accumulation
        #
        # Trainer still passes:
        #   [grad_accum, micro_batch, seq_len]
        # so for grad_accum == 1 we use batch[0].
        # ----------------------------------------------------

        if grad_accum == 1:
            micro_batch = batch[0]

            (
                (loss, _metrics),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            grads = jax.tree_util.tree_map(
                lambda g: g.astype(jnp.float32),
                grads,
            )

            return apply_update(
                state,
                params,
                grads,
                loss,
                batch,
            )

        # ----------------------------------------------------
        # Accumulation path: grad_accum > 1
        # ----------------------------------------------------

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(
                p,
                dtype=jnp.float32,
            ),
            params,
        )

        def scan_fn(
            grads_accum,
            micro_batch,
        ):
            (
                (loss, _metrics),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            grads_accum = jax.tree_util.tree_map(
                lambda acc, g: acc + g.astype(jnp.float32),
                grads_accum,
                grads,
            )

            return grads_accum, loss

        grads_accum, losses = jax.lax.scan(
            scan_fn,
            grads_accum,
            batch,
        )

        grads = jax.tree_util.tree_map(
            lambda g: g / jnp.asarray(
                grad_accum,
                dtype=jnp.float32,
            ),
            grads_accum,
        )

        loss = jnp.mean(
            losses,
            dtype=jnp.float32,
        )

        return apply_update(
            state,
            params,
            grads,
            loss,
            batch,
        )

    # ========================================================
    # PMAP compile
    # ========================================================

    return jax.pmap(
        train_step,
        axis_name="data",
        donate_argnums=(0,),
    )


# ============================================================
# Eval step
# ============================================================

def create_eval_step(
    *,
    model,
):

    def eval_step(
        state,
        batch,
    ):
        batch = constrain_batch(
            batch
        )

        inputs, targets = shift_tokens(
            batch
        )

        logits, _ = model.apply(
            {"params": state.params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, _ = compute_loss(
            logits,
            targets,
        )

        loss = jax.lax.pmean(
            loss.astype(jnp.float32),
            axis_name="data",
        )

        return {
            "loss": loss,
        }

    return jax.pmap(
        eval_step,
        axis_name="data",
    )