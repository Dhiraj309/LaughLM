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
# Train step with compiled gradient accumulation
# ------------------------------------------------------------

def create_train_step(
    model,
    optimizer: optax.GradientTransformation,
    grad_accum_steps: int,
) -> Callable:

    def loss_fn(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        loss, metrics = compute_loss(logits, targets)

        return loss, metrics


    def train_step(params: Params, opt_state: OptState, batch: Batch):

        # ----------------------------------------------------
        # reshape batch into microbatches for accumulation
        # ----------------------------------------------------
        B, T = batch.shape

        micro_batches = batch.reshape(
            grad_accum_steps,
            B // grad_accum_steps,
            T,
        )

        # ----------------------------------------------------
        # accumulation body
        # ----------------------------------------------------
        def accumulate(grads, micro_batch):

            (loss, metrics), step_grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params, micro_batch)

            grads = jax.tree_util.tree_map(
                lambda g, sg: g + sg,
                grads,
                step_grads,
            )

            return grads, (loss, metrics)

        # ----------------------------------------------------
        # initialize grad accumulator
        # ----------------------------------------------------
        grads_init = jax.tree_util.tree_map(
            jnp.zeros_like,
            params,
        )

        grads, outputs = jax.lax.scan(
            accumulate,
            grads_init,
            micro_batches,
        )

        # average gradients
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum_steps,
            grads,
        )

        # ----------------------------------------------------
        # optimizer update
        # ----------------------------------------------------
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        loss, metrics = outputs

        metrics["loss"] = loss

        return new_params, new_opt_state, metrics


    return jax.jit(train_step)


# ------------------------------------------------------------
# Evaluation step (unchanged from original)
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.jit(eval_step)