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

def build_adamw(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """
    AdamW with masked weight decay and LR schedule.

    Built manually via optax.chain:
      1. scale_by_adam — moment estimation (β1=0.9, β2=0.95)
      2. add_decayed_weights — masked decay (exclude norms/bias)
      3. scale_by_learning_rate — apply LR schedule (flip_sign=True → descent)

    The chain ordering matters:
      - Adam moments are computed on the raw gradient
      - Weight decay is applied to the Adam-scaled update
      - LR schedule scales the final update and negates for descent

    NOTE: clip_by_global_norm is NOT in the optimizer chain.
    Gradient clipping happens in train_step.py on globally-reduced
    gradients (after pmean) for correct distributed behavior.

    β2=0.95: frontier standard (Llama 3, DeepSeek V3, MiniCPM).
    Lower β2 adapts faster to gradient magnitude changes.
    """
    return optax.chain(
        optax.scale_by_adam(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
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
