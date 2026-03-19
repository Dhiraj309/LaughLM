import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict

from LaughLM.training.loss import shift_tokens, compute_loss


Params   = Any
OptState = Any
Batch    = jnp.ndarray
Metrics  = Dict[str, jnp.ndarray]


# ------------------------------------------------------------
# FUSED TRAIN STEP (scan-based)
# ------------------------------------------------------------

def create_train_step(model, optimizer, grad_accum: int) -> Callable:

    def loss_fn(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)

        loss, metrics = compute_loss(logits, targets)

        return loss, metrics


    def train_step(params: Params, opt_state: OptState, batch: Batch):

        """
        batch shape:
            [grad_accum, micro_batch, seq_len]
        """

        def micro_step(carry, micro_batch):
            params, opt_state = carry

            (loss, metrics), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params, micro_batch)

            return (params, opt_state), (grads, loss)


        # scan over micro-batches
        (_, _), (grads, losses) = jax.lax.scan(
            micro_step,
            (params, opt_state),
            batch,
        )

        # average gradients across micro-steps
        grads = jax.tree_util.tree_map(
            lambda g: jnp.mean(g, axis=0),
            grads,
        )

        # optimizer update
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        # average loss
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
