import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict

from LaughLM.training.loss import shift_tokens, compute_loss


Params = Any
OptState = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ------------------------------------------------------------
# TRUE GRADIENT ACCUMULATION (SCAN-BASED, MEMORY SAFE)
# ------------------------------------------------------------

def create_train_step(model, optimizer, grad_accum: int) -> Callable:

    def loss_fn(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss


    def train_step(params: Params, opt_state: OptState, batch: Batch):

        """
        batch shape:
            [grad_accum, micro_batch, seq_len]
        """

        # ------------------------------------------------------------
        # Initialize gradient accumulator
        # ------------------------------------------------------------
        grads_init = jax.tree_util.tree_map(jnp.zeros_like, params)


        # ------------------------------------------------------------
        # Micro-step (sequential, no vmap → no OOM)
        # ------------------------------------------------------------
        def micro_step(carry, micro_batch):
            params, grads_accum = carry

            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)

            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g,
                grads_accum,
                grads,
            )

            return (params, grads_accum), loss


        # ------------------------------------------------------------
        # Scan over micro-batches (TRUE accumulation)
        # ------------------------------------------------------------
        (_, grads), losses = jax.lax.scan(
            micro_step,
            (params, grads_init),
            batch,
        )


        # ------------------------------------------------------------
        # Average gradients
        # ------------------------------------------------------------
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads,
        )


        # ------------------------------------------------------------
        # Optimizer step
        # ------------------------------------------------------------
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        loss = jnp.mean(losses)

        metrics = {
            "loss": loss,
        }

        return new_params, new_opt_state, metrics


    return jax.jit(train_step)


# ------------------------------------------------------------
# EVAL STEP (unchanged)
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.jit(eval_step)
