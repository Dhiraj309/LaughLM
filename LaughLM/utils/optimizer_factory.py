"""
LaughLM/utils/optimizer_factory.py

Optax Optimizer Construction & Post-Reduction Gradient Clipping Pipeline.

Features:
1. Compressed AdamW Momentum State: Configure optax.adamw(..., mu_dtype=jnp.bfloat16) when
   optimizer_mu_bf16: true to store Adam's first moment (mu) in bfloat16, reducing total
   optimizer state memory consumption by ~33%.
2. Post-Reduction Gradient Clipping: Global norm gradient clipping is configured to execute
   strictly AFTER cross-replica gradient all-reduces across data-parallel ranks to ensure
   identical global norm calculation and mathematical synchronization across devices.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
import optax

from LaughLM.config.schema import LaughLMConfig
from LaughLM.training.optimizer import build_optimizer as build_native_optimizer, get_weight_decay_mask

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Optax AdamW Builder with Compressed Momentum (bf16 mu)
# ------------------------------------------------------------

def build_optimized_adamw(
    config: LaughLMConfig,
    schedule: Callable,
    optimizer_mu_bf16: bool = False,
) -> optax.GradientTransformation:
    """
    Build Optax AdamW optimizer chain with optional bfloat16 first-moment (mu) compression.

    When optimizer_mu_bf16=True, stores Adam's first moment (mu) in bfloat16,
    saving ~33% optimizer state memory.
    """
    mu_dtype = jnp.bfloat16 if optimizer_mu_bf16 else jnp.float32

    logger.info(
        f"[optimizer_factory] Building AdamW optimizer (mu_dtype={mu_dtype.__name__}, weight_decay={config.optimizer.weight_decay})."
    )

    return optax.chain(
        optax.scale_by_adam(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
            mu_dtype=mu_dtype,
        ),
        optax.add_decayed_weights(
            weight_decay=config.optimizer.weight_decay,
            mask=get_weight_decay_mask,
        ),
        optax.scale_by_learning_rate(schedule),
    )


# ------------------------------------------------------------
# Main Optimizer Factory Function
# ------------------------------------------------------------

def build_optimizer(
    config: LaughLMConfig,
    schedule: Callable,
) -> optax.GradientTransformation:
    """
    Build Optax optimizer dispatcher evaluating config.optimizations.optimizer_mu_bf16.
    """
    optimizer_mu_bf16 = getattr(
        getattr(config, "optimizations", None),
        "optimizer_mu_bf16",
        False,
    ) or (getattr(config.optimizer, "mu_dtype", "float32") == "bfloat16")

    opt_type = getattr(config.optimizer, "type", "adamw")

    if opt_type == "adamw":
        return build_optimized_adamw(
            config,
            schedule,
            optimizer_mu_bf16=optimizer_mu_bf16,
        )

    # Native fallbacks for adafactor, lion, etc.
    return build_native_optimizer(config, schedule)


# ------------------------------------------------------------
# Post-Reduction Gradient Clipping
# ------------------------------------------------------------

def clip_gradients_post_reduction(
    grads: Any,
    max_norm: float,
) -> Tuple[Any, jnp.ndarray]:
    """
    Apply global norm gradient clipping strictly AFTER cross-replica all-reduces.

    Returns:
        (clipped_grads, global_norm)
    """
    if max_norm <= 0.0:
        global_norm = optax.global_norm(grads)
        return grads, global_norm

    # Global norm over unclipped reduced gradients
    global_norm = optax.global_norm(grads)

    # Scaling factor = min(1.0, max_norm / (global_norm + 1e-6))
    scaling = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))

    clipped_grads = jax.tree_util.tree_map(
        lambda g: g * scaling if g is not None else None,
        grads,
    )

    return clipped_grads, global_norm
