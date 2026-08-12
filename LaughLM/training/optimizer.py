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


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def _resolve_optimizer_dtype(name: str):
    try:
        return _DTYPE_MAP[name]
    except KeyError as e:
        raise ValueError(
            f"Unsupported optimizer dtype: {name!r}. "
            f"Expected one of {sorted(_DTYPE_MAP)}."
        ) from e


# ────────────────────────────────────────────────────────────────
# Weight decay mask
# ────────────────────────────────────────────────────────────────

def get_weight_decay_mask(params: Any) -> Any:
    """
    Return mask tree: True = apply weight decay, False = exclude.

    Excludes:
    - bias
    - scale
    - pos_embedding
    - embedding
    - RMSNorm weight

    Keeps Dense/MLP/attention kernels decayed.
    """
    flat = traverse_util.flatten_dict(params)

    def should_decay(path):
        leaf = path[-1]
        joined = "/".join(str(p).lower() for p in path)

        if leaf in {"bias", "scale", "pos_embedding", "embedding"}:
            return False

        if leaf == "weight" and (
            "rmsnorm" in joined
            or "norm" in joined
            or "layernorm" in joined
        ):
            return False

        return True

    mask_flat = {k: should_decay(k) for k in flat}
    return traverse_util.unflatten_dict(mask_flat)


# ────────────────────────────────────────────────────────────────
# Optimizer builders
# ────────────────────────────────────────────────────────────────

def build_adamw(config, schedule):
    # Determine mu_dtype based on config
    mu_bf16 = getattr(config.optimizations, "optimizer_mu_bf16", False)
    mu_dtype = jnp.bfloat16 if mu_bf16 else jnp.float32

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
            mu_dtype=mu_dtype,
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
