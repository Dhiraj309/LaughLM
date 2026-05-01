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
# TRAIN STEP (PMAP)
# ------------------------------------------------------------

def create_train_step(model, optimizer, grad_accum: int, axis_name="batch") -> Callable:

    # --------------------------------------------------------
    # Loss function (per micro-batch)
    # --------------------------------------------------------
    def loss_fn(params: Params, micro_batch: Batch):
        # 🔴 CRITICAL: enforce shape contract
        assert micro_batch.ndim == 2, f"[loss_fn] Expected (B, T), got {micro_batch.shape}"

        inputs, targets = shift_tokens(micro_batch)

        # Safety check after shift
        assert inputs.ndim == 2, f"[loss_fn] inputs shape broken: {inputs.shape}"

        logits = model.apply({"params": params}, inputs)

        loss, _ = compute_loss(logits, targets)
        return loss

    # --------------------------------------------------------
    # Train step (per device)
    # --------------------------------------------------------
    def train_step(state, batch):
        """
        batch shape inside pmap:
            (grad_accum, micro_batch, seq_len)
        """

        # 🔴 CRITICAL: validate batch shape early
        assert batch.ndim == 3, f"[train_step] Expected (grad_accum, micro_batch, seq), got {batch.shape}"

        params = state.params
        opt_state = state.opt_state

        # --------------------------------------------------------
        # RNG
        # --------------------------------------------------------
        state, step_rng = state.next_rng()

        # --------------------------------------------------------
        # INIT gradient accumulator (FP32 for stability)
        # --------------------------------------------------------
        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params,
        )

        # --------------------------------------------------------
        # SCAN over micro-batches
        # --------------------------------------------------------
        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry

            # 🔴 enforce correct shape INSIDE scan
            assert micro_batch.ndim == 2, f"[scan_fn] micro_batch wrong shape: {micro_batch.shape}"

            rng, subkey = jax.random.split(rng)

            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                micro_batch,
            )

            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g,
                grads_accum,
                grads,
            )

            return (grads_accum, rng), loss

        (grads_accum, _), losses = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
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
        # GRAD NORM (critical for debugging + stability)
        # --------------------------------------------------------
        grad_norm = optax.global_norm(grads)
        grad_norm = jax.lax.pmean(grad_norm, axis_name)

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        # --------------------------------------------------------
        # TOKEN ACCOUNTING
        # --------------------------------------------------------
        tokens_in_step = (
            batch.shape[0]  # grad_accum
            * batch.shape[1]  # micro_batch
            * batch.shape[2]  # seq_len
        )

        # --------------------------------------------------------
        # STATE UPDATE
        # --------------------------------------------------------
        new_state = state.apply_grad_step(
            new_params,
            new_opt_state,
            tokens_in_step,
        )

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
        }

        return new_state, metrics

    # ------------------------------------------------------------
    # PMAP
    # ------------------------------------------------------------
    return jax.pmap(
        train_step,
        axis_name=axis_name,
        donate_argnums=(0,),
    )


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):
        assert batch.ndim == 2, f"[eval_step] Expected (B, T), got {batch.shape}"

        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)
        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.pmap(eval_step, axis_name="batch")
