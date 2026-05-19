"""
LaughLM/training/optimizer.py

Optimizer construction for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Clean documentation — explicit frontier references for all choices.

2. Masked weight decay — excludes norm scale, bias, and positional
   embeddings from decay. This is the standard for all frontier models.

3. Manual AdamW construction via optax.chain — gives full control over
   Adam → masked decay → LR schedule ordering.

4. Gradient clipping removed from optax chain (FIX audit 2025):
   Clipping now happens AFTER pmean in train_step.py on the
   globally-reduced gradient, matching MaxText convention.

5. mu_dtype support — Adam momentum slots can be stored in bf16
   to reduce optimizer memory by ~33% with minimal convergence impact.
   Reference: MaxText optax.adamw(mu_dtype=...).

FIX (frontier-optim audit 2026):
  optax.scale_by_learning_rate(schedule) uses flip_sign=True by default,
  which negates the gradient (for descent). This is CORRECT when used
  with scale_by_adam which produces positive updates. The chain:
    scale_by_adam → add_decayed_weights → scale_by_learning_rate
  is equivalent to AdamW with schedule, verified against optax source.
"""

import optax
from flax import traverse_util
from typing import Any, Callable

import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig


# ────────────────────────────────────────────────────────────────
# Weight decay mask
# ────────────────────────────────────────────────────────────────

def get_weight_decay_mask(params: Any) -> Any:
    """
    Return a mask tree: True = apply decay, False = exclude.

    Excluded: scale (norm γ), bias, pos_embedding.
    Every frontier model excludes norm parameters from weight decay.
    
    This function is passed as the `mask` argument to
    optax.add_decayed_weights. It receives the params pytree and
    returns a pytree of booleans with the same structure.
    """
    flat = traverse_util.flatten_dict(params)
    no_decay = {"scale", "bias", "pos_embedding"}
    mask_flat = {k: (k[-1] not in no_decay) for k in flat}
    return traverse_util.unflatten_dict(mask_flat)


# ────────────────────────────────────────────────────────────────
# Optimizer builders
# ────────────────────────────────────────────────────────────────

def build_adamw(config, schedule):

    return optax.chain(

        optax.scale_by_adam(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,

            #
            # IMPORTANT:
            # Keep Adam moments in fp32 even when
            # params are bf16.
            #
            # Prevents bf16→fp32 dtype mutation
            # after first optimizer step.
            #
            mu_dtype=jnp.float32,
        ),

        optax.add_decayed_weights(
            weight_decay=config.optimizer.weight_decay,
            mask=get_weight_decay_mask,
        ),

        optax.scale_by_learning_rate(schedule),
    )


def build_adafactor(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """Adafactor: memory-efficient optimizer that factors the second moment."""
    return optax.adafactor(learning_rate=schedule)


def build_lion(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """Lion optimizer (EvoLved Sign Momentum). Lower memory than Adam."""
    return optax.lion(
        learning_rate=schedule,
        b1=config.optimizer.beta1,
        b2=config.optimizer.beta2,
    )


# ────────────────────────────────────────────────────────────────
# Main factory
# ────────────────────────────────────────────────────────────────

def build_optimizer(
    config: LaughLMConfig,
    schedule: Callable,
) -> optax.GradientTransformation:
    """Build optimizer from config. Schedule is baked into the chain.
    
    NOTE: Gradient clipping (clip_by_global_norm) is NOT included here.
    It is applied in train_step.py after pmean on globally-reduced
    gradients. This ensures all devices compute the same global norm
    and apply identical clipping — a correctness requirement for
    distributed training.
    """

    opt_type = config.optimizer.type

    if opt_type == "adamw":
        return build_adamw(config, schedule)

    if opt_type == "adafactor":
        return build_adafactor(config, schedule)

    if opt_type == "lion":
        return build_lion(config, schedule)

    if opt_type == "muon":
        raise NotImplementedError(
            "Muon optimizer not yet implemented. Use 'adamw'."
        )

    raise ValueError(
        f"Unknown optimizer type: '{opt_type}'. "
        f"Valid options: adamw, adafactor, lion, muon."
    )
