import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict, Tuple

from LaughLM.training.loss import shift_tokens, compute_loss


Params   = Any
OptState = Any
Batch    = jnp.ndarray
Metrics  = Dict[str, jnp.ndarray]


# ------------------------------------------------------------
# Forward + backward (no optimizer step)
# ------------------------------------------------------------

def create_train_step(model) -> Callable:

    def loss_fn(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        loss, metrics = compute_loss(logits, targets)

        return loss, metrics


    def train_step(params: Params, batch: Batch):

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params, batch)

        metrics["loss"] = loss

        return grads, metrics


    return jax.jit(train_step)


# ------------------------------------------------------------
# Optimizer application
# ------------------------------------------------------------

def apply_optimizer(
    optimizer: optax.GradientTransformation,
) -> Callable:

    def step(params, opt_state, grads):

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    return jax.jit(step)


# ------------------------------------------------------------
# Evaluation step
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.jit(eval_step)
