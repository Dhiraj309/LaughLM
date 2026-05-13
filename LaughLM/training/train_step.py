"""
LaughLM/training/train_step.py

pmap training step with correct distributed gradient semantics.

FRONTIER-GRADE (2026):
- Float32 gradient accumulation (mandatory for bf16 training)
- Gradient clipping AFTER pmean (correct optimization semantics)
- Decode-aware: training step only - decode uses separate path
- Static shapes throughout (no recompilation)
- Zero host synchronization in hot path

CORRECTNESS NOTE on gradient clipping order:
  Clipping before pmean does NOT cause replica divergence - after pmean,
  all devices still converge to the same averaged result. However, it
  changes OPTIMIZATION SEMANTICS:
    clip(g_i) != clip(mean(g_i))
  Clipping local gradients biases updates toward smaller local norms.
  Frontier systems (MaxText, Pax, Megatron, DeepSpeed) all perform:
    grads = pmean(grads) -> grad_norm = global_norm(grads) -> clip(grads)
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

        Execution order:
          1. Accumulate gradients in fp32 (per-device, via lax.scan)
          2. Average across accumulation steps
          3. pmean across devices (all-reduce)
          4. Compute global norm and clip (on globally-correct gradient)
          5. Optimizer step
          6. Update state
        """

        params = state.params
        opt_state = state.opt_state
        state, step_rng = state.next_rng()

        # Gradient accumulator - ALWAYS float32
        # bf16 accumulation loses mantissa precision over microbatches
        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.float32),
            params,
        )

        def scan_fn(carry, micro_batch):
            grads_accum, rng = carry
            rng, subkey = jax.random.split(rng)
            loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)
            grads_accum = jax.tree_util.tree_map(
                lambda g_acc, g: g_acc + g.astype(jnp.float32),
                grads_accum, grads,
            )
            return (grads_accum, rng), loss

        (grads_accum, _), losses = jax.lax.scan(
            scan_fn, (grads_accum, step_rng), batch,
        )

        grads = jax.tree_util.tree_map(lambda g: g / grad_accum, grads_accum)
        loss = jnp.mean(losses)

        # Cross-device sync FIRST - all devices get identical gradients
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # Global clip AFTER pmean (correct semantics)
        grad_norm = optax.global_norm(grads)
        clip_scale = jnp.minimum(1.0, max_grad_norm / jnp.maximum(grad_norm, 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        # Optimizer step
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Per-device token accounting (no jax.device_count inside pmap)
        tokens_in_step = batch.shape[0] * batch.shape[1] * batch.shape[2]

        new_state = state.apply_grad_step(new_params, new_opt_state, tokens_in_step)

        return new_state, {"loss": loss, "grad_norm": grad_norm}

    return jax.pmap(train_step, axis_name=axis_name, donate_argnums=(0,))


def create_eval_step(model) -> Callable:
    def eval_step(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits, _ = model.apply({"params": params}, inputs)
        _, metrics = compute_loss(logits, targets)
        return metrics
    return jax.pmap(eval_step, axis_name="batch")
