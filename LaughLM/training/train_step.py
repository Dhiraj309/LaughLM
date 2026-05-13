"""
LaughLM/training/train_step.py

pmap training step with correct distributed gradient semantics.

FRONTIER-GRADE (2026):
──────────────────────────────────────────────
- Float32 gradient accumulation (mandatory for bf16 training)
- Gradient clipping AFTER pmean (correct optimization semantics)
- Decode-aware: training step only — decode uses separate path
- Static shapes throughout (no recompilation)
- Zero host synchronization in hot path

CORRECTNESS NOTE on gradient clipping order:
  Clipping before pmean does NOT cause replica divergence — after pmean,
  all devices still converge to the same averaged result. However, it
  changes OPTIMIZATION SEMANTICS:
    clip(g_i) != clip(mean(g_i))
  Clipping local gradients biases updates toward smaller local norms.
  Frontier systems (MaxText, Pax, Megatron, DeepSpeed) all perform:
    grads = pmean(grads) → grad_norm = global_norm(grads) → clip(grads)
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
    """
    Create a pmap-wrapped training step.

    The returned function has signature:
        (state, batch) -> (new_state, metrics)

    where batch shape (inside pmap, device axis stripped):
        (grad_accum, micro_batch_per_device, seq_len)

    Static shapes are enforced throughout to prevent recompilation.
    """

    def loss_fn(params: Params, micro_batch: Batch):
        inputs, targets = shift_tokens(micro_batch)
        logits, _ = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss

    def train_step(state, batch):
        """
        Single optimizer step with gradient accumulation.

        Execution order:
          1. Accumulate gradients in fp32 (per-device, via lax.scan)
          2. Average across accumulation steps
          3. pmean across devices (all-reduce)
          4. Compute global norm and clip (on globally-correct gradient)
          5. Optimizer step
          6. Update state

        This ordering ensures all devices compute identical updates.
        """

        params = state.params
        opt_state = state.opt_state

        # RNG split for this step
        state, step_rng = state.next_rng()

        # ─────────────────────────────────────────────
        # Gradient accumulator — ALWAYS float32
        #
        # bf16 accumulation loses mantissa precision over multiple
        # microbatches, causing noisy gradients and worse convergence.
        # This is mandatory for frontier training stability.
        # Reference: MaxText, Megatron-LM, DeepSpeed all use fp32.
        # ─────────────────────────────────────────────

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params,
        )

        # ─────────────────────────────────────────────
        # Gradient accumulation via lax.scan
        # (canonical JAX pattern — confirmed by MaxText)
        # ─────────────────────────────────────────────

        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry
            rng, subkey = jax.random.split(rng)

            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)

            # Accumulate in float32 regardless of param dtype
            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g.astype(jnp.float32),
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
        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum, grads_accum
        )
        loss = jnp.mean(losses)

        # ─────────────────────────────────────────────
        # Cross-device synchronization (all-reduce)
        #
        # After pmean, all devices have identical gradients.
        # This MUST happen before clipping to get correct
        # global norm semantics: clip(mean(g_i)) not mean(clip(g_i)).
        # ─────────────────────────────────────────────

        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # ─────────────────────────────────────────────
        # Global gradient clipping AFTER pmean
        #
        # Now all devices have identical grads → identical global
        # norm → identical clipping → identical optimizer updates.
        #
        # This is the correct optimization semantics. Clipping before
        # pmean biases toward smaller local norms, changing the
        # effective learning signal.
        # ─────────────────────────────────────────────

        grad_norm = optax.global_norm(grads)

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm / jnp.maximum(grad_norm, 1e-8),
        )

        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        # ─────────────────────────────────────────────
        # Optimizer step (fp32 gradients → optimizer)
        # ─────────────────────────────────────────────

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # ─────────────────────────────────────────────
        # Per-device token accounting
        #
        # Inside pmap, device axis is stripped. batch.shape =
        # (grad_accum, micro_batch_per_device, seq_len).
        # Global tokens assembled in trainer.py.
        # ─────────────────────────────────────────────

        tokens_in_step = (
            batch.shape[0]   # grad_accum
            * batch.shape[1] # micro_batch_per_device
            * batch.shape[2] # seq_len
        )

        # ─────────────────────────────────────────────
        # State update (no host sync here)
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
    """Create pmap-wrapped evaluation step (no gradient computation)."""

    def eval_step(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits, _ = model.apply({"params": params}, inputs)
        _, metrics = compute_loss(logits, targets)
        return metrics

    return jax.pmap(eval_step, axis_name="batch")
