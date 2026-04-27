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
# TRUE GRADIENT ACCUMULATION (SCAN-BASED, OPTIMIZED)
# ------------------------------------------------------------

def create_train_step(model, optimizer, grad_accum: int, axis_name="batch") -> Callable:

    def loss_fn(params: Params, micro_batch: Batch):
        """
        micro_batch shape: (micro_batch_size, seq_len)
        """
        inputs, targets = shift_tokens(micro_batch)
        logits = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss

    def train_step(params: Params, opt_state: OptState, batch: Batch):
        """
        batch shape (AFTER trainer swapaxes):
            (num_devices, grad_accum, micro_batch, seq_len)

        Inside pmap each device sees:
            (grad_accum, micro_batch, seq_len)
        """

        # --------------------------------------------------------
        # INIT gradient accumulator
        # --------------------------------------------------------
        grads_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

        # --------------------------------------------------------
        # SCAN over micro-batches (true accumulation)
        # --------------------------------------------------------
        def scan_fn(carry, micro_batch):
            grads_accum = carry

            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)

            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g,
                grads_accum,
                grads,
            )

            return grads_accum, loss

        grads_accum, losses = jax.lax.scan(
            scan_fn,
            grads_accum,
            batch,
        )

        # --------------------------------------------------------
        # AVERAGE gradients
        # --------------------------------------------------------
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(losses)

        # --------------------------------------------------------
        # CROSS-DEVICE SYNC
        # --------------------------------------------------------
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------
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

    # ------------------------------------------------------------
    # PMAP (multi-device)
    # ------------------------------------------------------------
    return jax.pmap(
        train_step,
        axis_name=axis_name,
        donate_argnums=(0, 1),
    )


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):

        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)
        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.pmap(eval_step, axis_name="batch")