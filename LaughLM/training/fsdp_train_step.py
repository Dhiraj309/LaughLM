"""
LaughLM/training/fsdp_train_step.py

GSPMD/FSDP train step.

No pmap.
No lax.pmean.
Global array semantics + shardings handle collectives.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from LaughLM.training.loss import shift_tokens, compute_loss
from LaughLM.distributed.sharding import constrain_batch, constrain_logits


Metrics = Dict[str, jnp.ndarray]


def create_fsdp_train_step(
    *,
    model,
    optimizer,
    state_sharding,
    batch_sharding,
    metrics_sharding,
    grad_accum: int,
    max_grad_norm: float = 1.0,
):

    def loss_fn(params, micro_batch):
        inputs, targets = shift_tokens(micro_batch)

        logits, _ = model.apply(
            {"params": params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
        )

        logits = constrain_logits(logits)

        loss, metrics = compute_loss(
            logits,
            targets,
        )

        return loss, metrics

    def train_step(state, batch):
        batch = constrain_batch(batch)

        params = state.params

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params,
        )

        step_rng = jax.random.fold_in(
            state.rng_key,
            state.step,
        )

        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry
            rng, _ = jax.random.split(rng)

            (loss, _metrics), grads = jax.value_and_grad(
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

            return (grads_accum, rng), loss

        (grads_accum, _), losses = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
            batch,
        )

        grads = jax.tree_util.tree_map(
            lambda g: g / jnp.asarray(grad_accum, dtype=jnp.float32),
            grads_accum,
        )

        loss = jnp.mean(losses, dtype=jnp.float32)

        grad_norm = optax.global_norm(grads).astype(jnp.float32)

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm / jnp.maximum(grad_norm, 1e-6),
        )

        grads = jax.tree_util.tree_map(
            lambda g: g * clip_scale,
            grads,
        )

        updates, new_opt_state = optimizer.update(
            grads,
            state.opt_state,
            params,
        )

        new_params = optax.apply_updates(
            params,
            updates,
        )

        tokens_in_step = jnp.asarray(
            batch.shape[0] * batch.shape[1] * batch.shape[2],
            dtype=jnp.int32,
        )

        new_state = state.apply_grad_step(
            params=new_params,
            opt_state=new_opt_state,
            tokens_in_step=tokens_in_step,
        )

        metrics = {
            "loss": loss.astype(jnp.float32),
            "grad_norm": grad_norm.astype(jnp.float32),
        }

        return new_state, metrics

    return jax.jit(
        train_step,
        in_shardings=(state_sharding, batch_sharding),
        out_shardings=(state_sharding, metrics_sharding),
        donate_argnums=(0,),
    )