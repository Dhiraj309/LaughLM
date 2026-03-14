import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict, Tuple

from LaughLM.training.loss import shift_tokens, compute_loss


Params   = Any
OptState = Any
Batch    = jnp.ndarray
Metrics  = Dict[str, jnp.ndarray]


def create_train_step(
    model,
    optimizer: optax.GradientTransformation,
) -> Callable:

    def loss_fn(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        loss, metrics = compute_loss(logits, targets)

        return loss, metrics

    def train_step(params: Params, opt_state: OptState, batch: Batch):

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params, batch)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)

        # Keep everything as JAX scalars inside JIT
        metrics["loss"] = loss

        return new_params, new_opt_state, metrics

    return jax.jit(train_step)


def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.jit(eval_step)
