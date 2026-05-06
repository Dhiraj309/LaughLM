"""
LaughLM/training/train_step.py

pmap training step with proper gradient all-reduce.

FINAL (2026):
──────────────────────────────────────────────
- Correct global token accounting (includes all devices)
- Safe for multi-device / multi-host setups
- BF16-friendly gradient accumulation
- Per-device grad clipping BEFORE pmean
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


def create_train_step(
    model,
    optimizer,
    grad_accum: int,
    max_grad_norm: float = 1.0,
    axis_name="batch",
) -> Callable:

    def loss_fn(params: Params, micro_batch: Batch):
        inputs, targets = shift_tokens(micro_batch)
        logits, _ = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss

    def train_step(state, batch):
        """
        batch shape inside pmap:
        (grad_accum, micro_batch_per_device, seq_len)
        """

        params = state.params
        opt_state = state.opt_state

        # RNG
        state, step_rng = state.next_rng()

        # ─────────────────────────────────────────────
        # Gradient accumulator (same dtype as params)
        # ─────────────────────────────────────────────

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=p.dtype),
            params,
        )

        # ─────────────────────────────────────────────
        # Gradient accumulation loop
        # ─────────────────────────────────────────────

        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry
            rng, subkey = jax.random.split(rng)

            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)

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

        # Average grads across accumulation steps
        grads = jax.tree_util.tree_map(lambda g: g / grad_accum, grads_accum)
        loss = jnp.mean(losses)

        # ─────────────────────────────────────────────
        # Gradient clipping (per-device BEFORE pmean)
        # ─────────────────────────────────────────────

        grad_norm = optax.global_norm(grads)

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm / jnp.maximum(grad_norm, 1e-8),
        )

        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        # ─────────────────────────────────────────────
        # Cross-device synchronization
        # ─────────────────────────────────────────────

        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)
        grad_norm = jax.lax.pmean(grad_norm, axis_name)

        # ─────────────────────────────────────────────
        # Optimizer step (fp32 master weights)
        # ─────────────────────────────────────────────

        grads = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # ─────────────────────────────────────────────
        # ✅ Correct GLOBAL token accounting
        # ─────────────────────────────────────────────

        tokens_in_step = (
            batch.shape[0]   # grad_accum
            * batch.shape[1] # micro_batch_per_device
            * batch.shape[2] # seq_len
            * jax.device_count()  # 🔥 correct global device count
        )

        # ─────────────────────────────────────────────
        # State update
        # ─────────────────────────────────────────────

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


def create_eval_step(model) -> Callable:

    def eval_step(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits, _ = model.apply({"params": params}, inputs)
        _, metrics = compute_loss(logits, targets)
        return metrics

    return jax.pmap(eval_step, axis_name="batch")