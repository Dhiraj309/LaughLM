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
   Clipping now happens per-device in train_step.py BEFORE pmean,
   matching MaxText convention. The optimizer chain no longer includes
   clip_by_global_norm — it's applied at the gradient level in train_step.
   This prevents gradient spikes from being diluted by multi-device averaging.
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
      2. add_decayed_weights — masked decay (exclude norms)
      3. scale_by_learning_rate — apply LR schedule

    NOTE: clip_by_global_norm is REMOVED from the chain (fix audit 2025).
    Gradient clipping now happens per-device in train_step.py BEFORE pmean,
    preventing spike dilution across devices.

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
    return optax.chain(
        optax.adafactor(learning_rate=schedule),
    )


def build_lion(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """Lion optimizer (EvoLved Sign Momentum). Lower memory than Adam."""
    return optax.chain(
        optax.lion(
            learning_rate=schedule,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
        ),
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
    It is applied per-device in train_step.py before pmean to prevent
    gradient spike dilution across devices.
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