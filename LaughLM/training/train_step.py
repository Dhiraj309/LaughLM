"""
LaughLM/training/train_step.py

Training step for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Updated for new model signature — GPTModel now returns
   (logits, kv_caches) tuple. Training ignores caches.

2. Removed assert statements inside jitted code — JAX traces
   through asserts which causes ConcretizationError. Replaced
   with shape comments for documentation.

3. Gradient accumulation via jax.lax.scan — unchanged, already
   correct. FP32 gradient accumulators for numerical stability.

Note: This file still uses jax.pmap for now. The SPMD/jit migration
(replacing pmap with jax.jit + mesh sharding) will happen when
trainer.py is updated to construct the mesh. Both pmap and SPMD
can coexist — pmap is a special case of SPMD.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict

from LaughLM.training.loss import shift_tokens, compute_loss


Params = Any
OptState = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ────────────────────────────────────────────────────────────────
# TRAIN STEP (PMAP)
# ────────────────────────────────────────────────────────────────

def create_train_step(model, optimizer, grad_accum: int, axis_name="batch") -> Callable:

    def loss_fn(params: Params, micro_batch: Batch):
        # micro_batch: (B, T) — integer token IDs
        inputs, targets = shift_tokens(micro_batch)

        # Model returns (logits, kv_caches) — ignore caches during training
        logits, _ = model.apply({"params": params}, inputs)

        loss, _ = compute_loss(logits, targets)
        return loss

    def train_step(state, batch):
        """
        batch shape inside pmap: (grad_accum, micro_batch, seq_len)
        """
        params = state.params
        opt_state = state.opt_state

        # RNG
        state, step_rng = state.next_rng()

        # Init gradient accumulator (FP32 for stability)
        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params,
        )

        # Scan over micro-batches
        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry
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

        # Average gradients
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(losses)

        # Cross-device sync
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # Grad norm (for debugging + stability)
        grad_norm = optax.global_norm(grads)
        grad_norm = jax.lax.pmean(grad_norm, axis_name)

        # Optimizer step
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        new_params = optax.apply_updates(params, updates)

        # Token accounting
        tokens_in_step = (
            batch.shape[0]   # grad_accum
            * batch.shape[1]  # micro_batch
            * batch.shape[2]  # seq_len
        )

        # State update
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

    return jax.pmap(
        train_step,
        axis_name=axis_name,
        donate_argnums=(0,),
    )


# ────────────────────────────────────────────────────────────────
# EVAL STEP
# ────────────────────────────────────────────────────────────────

def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):
        # batch: (B, T)
        inputs, targets = shift_tokens(batch)

        # Model returns (logits, kv_caches)
        logits, _ = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.pmap(eval_step, axis_name="batch")
