"""
LaughLM/training/train_step.py
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import optax

from flax.linen import (
    partitioning as nn_partitioning,
)

from LaughLM.training.loss import (
    shift_tokens,
    compute_loss,
)

from LaughLM.distributed.sharding import (
    get_logical_axis_rules,
    constrain_batch,
    constrain_logits,
    replicated_sharding,
)


Params = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ─────────────────────────────────────────────────────────────
# Train step
# ─────────────────────────────────────────────────────────────

def create_train_step(
    *,
    model,
    optimizer,
    config,
    mesh,
    state_shardings,
    data_sharding,
    grad_accum,
    max_grad_norm=1.0,
):

    metrics_sharding = (
        replicated_sharding(mesh)
    )

    def loss_fn(
        params,
        micro_batch,
    ):

        inputs, targets = shift_tokens(
            micro_batch
        )

        logits, _ = model.apply(
            {"params": params},
            inputs,
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

    def train_step(
        state,
        batch,
    ):

        batch = constrain_batch(
            batch
        )

        params = state.params

        step_rng = jax.random.fold_in(
            state.rng_key,
            state.step,
        )

        grads_accum = (
            jax.tree_util.tree_map(
                lambda p: jnp.zeros_like(
                    p,
                    dtype=jnp.float32,
                ),
                params,
            )
        )

        def scan_fn(
            carry,
            micro_batch,
        ):

            grads_accum, rng = carry

            rng, _ = jax.random.split(
                rng
            )

            (
                (loss, metrics),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            grads_accum = (
                jax.tree_util.tree_map(
                    lambda a, g:
                    a + g.astype(jnp.float32),
                    grads_accum,
                    grads,
                )
            )

            return (
                (grads_accum, rng),
                loss,
            )

        (
            (grads_accum, _),
            losses,
        ) = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
            batch,
        )

        grads = jax.tree_util.tree_map(
            lambda g:
            g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(
            losses,
            dtype=jnp.float32,
        )

        grad_norm = optax.global_norm(
            grads
        )

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

        updates, new_opt_state = (
            optimizer.update(
                grads,
                state.opt_state,
                params,
            )
        )

        new_params = (
            optax.apply_updates(
                params,
                updates,
            )
        )

        tokens_in_step = int(
            batch.shape[0]
            * batch.shape[1]
            * batch.shape[2]
        )

        new_state = (
            state.apply_grad_step(
                params=new_params,
                opt_state=new_opt_state,
                tokens_in_step=tokens_in_step,
            )
        )

        metrics = {
            "loss": loss.astype(
                jnp.float32
            ),
            "grad_norm": grad_norm.astype(
                jnp.float32
            ),
        }

        return (
            new_state,
            metrics,
        )

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        return jax.jit(
            train_step,

            in_shardings=(
                state_shardings,
                data_sharding,
            ),

            out_shardings=(
                state_shardings,
                metrics_sharding,
            ),

            donate_argnums=(0,),
        )


# ─────────────────────────────────────────────────────────────
# Eval step
# ─────────────────────────────────────────────────────────────

def create_eval_step(
    *,
    model,
    config,
    mesh,
    state_shardings,
    data_sharding,
):

    metrics_sharding = (
        replicated_sharding(mesh)
    )

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
            inputs,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, _ = compute_loss(
            logits,
            targets,
        )

        return {
            "loss": loss.astype(
                jnp.float32
            ),
        }

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        return jax.jit(
            eval_step,

            in_shardings=(
                state_shardings,
                data_sharding,
            ),

            out_shardings=metrics_sharding,
        )
