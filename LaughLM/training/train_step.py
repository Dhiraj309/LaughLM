
import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict, Tuple

from LaughLM.training.loss import shift_tokens, compute_loss


# ------------------------------------------------------------
# Type aliases
# ------------------------------------------------------------

Params   = Any
OptState = Any
Batch    = jnp.ndarray   # [B, T] integer token IDs
Metrics  = Dict[str, float]


# ------------------------------------------------------------
# Train step factory
# ------------------------------------------------------------

def create_train_step(
    model,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """
    Build and JIT-compile a single gradient update step.

    Returns a function with signature:
        train_step(params, opt_state, batch) → (params, opt_state, metrics)

    Design notes
    ------------
    - jax.value_and_grad computes both the loss value and its gradient
      in a single forward+backward pass. has_aux=True means loss_fn returns
      (loss_scalar, auxiliary_data) — the gradient is taken wrt loss_scalar
      and auxiliary_data is returned unchanged.

    - jax.jit traces the function and compiles it to XLA at first call.
      Subsequent calls reuse the compiled version — no recompilation.
      On TPU, jit is mandatory for reasonable performance.

    - optax.apply_updates adds the optimizer's update (already sign-flipped
      by optax.scale_by_learning_rate) to the current parameters.
    """

    def loss_fn(
        params: Params,
        batch: Batch,
    ) -> Tuple[jnp.ndarray, Tuple[Metrics, jnp.ndarray]]:
        """
        Forward pass + loss computation.
        Returns (scalar_loss, (metrics_dict, logits)) for value_and_grad.
        """
        # Shift: inputs = tokens[:-1], targets = tokens[1:]
        inputs, targets = shift_tokens(batch)

        # Forward pass
        logits = model.apply({"params": params}, inputs)  # [B, T-1, V]

        # Loss (scalar) + metrics dict
        loss, metrics = compute_loss(logits, targets)

        return loss, (metrics, logits)

    # ----------------------------------------------------------------
    # FIX 1: function renamed 'train_step' (was 'train_state' — mismatch
    #         with jax.jit(train_step) below which used undefined name)
    # FIX 2: 'jax.grad_and_value' → 'jax.value_and_grad'
    #         (JAX API is value_and_grad; grad_and_value does not exist)
    # FIX 3: return order is (value, grads) not (grads, value)
    # FIX 4: 'out_state' → 'new_opt_state' (consistent naming)
    # ----------------------------------------------------------------

    def train_step(
        params: Params,
        opt_state: OptState,
        batch: Batch,
    ) -> Tuple[Params, OptState, Metrics]:

        # Compute loss value AND gradients in one pass
        # value_and_grad with has_aux=True:
        #   returns ((loss, aux), grads)  where aux = (metrics, logits)
        (loss, (metrics, logits)), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params, batch)

        # Compute optimizer update (includes gradient clipping + LR schedule
        # because these are baked into the optimizer chain in optimizer.py)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)

        # Apply updates: new_params = params + updates
        # (updates are already negated by optax.scale_by_learning_rate)
        new_params = optax.apply_updates(params, updates)

        # Add scalar loss to metrics for logging
        metrics["loss"] = float(loss)

        return new_params, new_opt_state, metrics

    # JIT-compile for XLA (mandatory on TPU, large speedup on GPU)
    return jax.jit(train_step)


# ------------------------------------------------------------
# Eval step (no gradient computation)
# ------------------------------------------------------------

def create_eval_step(model) -> Callable:
    """
    Build a JIT-compiled evaluation step (no gradient, no optimizer update).
    Used to compute validation loss during training.
    """

    def eval_step(params: Params, batch: Batch) -> Metrics:

        inputs, targets = shift_tokens(batch)

        logits = model.apply({"params": params}, inputs)

        _, metrics = compute_loss(logits, targets)

        return metrics

    return jax.jit(eval_step)
