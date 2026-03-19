
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
# FUSED TRAIN STEP (scan-based)
# ------------------------------------------------------------


def create_train_step(model, optimizer, grad_accum: int) -> Callable:

    def loss_on_batch(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss


    def train_step(params: Params, opt_state: OptState, batch: Batch):

        """
        batch shape:
            [grad_accum, micro_batch, seq_len]
        """

        # ------------------------------
------------------------------

        # Flatten batch → avoid scan/JVP
        # ------------------------------
------------------------------


        batch = batch.reshape(
            batch.shape[0] * batch.shape[1],
            batch.shape[2],
        )

        # ------------------------------
------------------------------

        # Single backward pass (CRITICAL)
        # ------------------------------
------------------------------

        def loss_fn(params):
            losses = jax.vmap(lambda b: loss_on_batch(params, b[None, :]))(batch)

            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(
params)

        # ------------------------------------------------------------

        # Optimizer step
        # ------------------------------
------------------------------


        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

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
