import optax
from flax import traverse_util
from typing import Any, Callable

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Weight decay mask
# ------------------------------------------------------------

def get_weight_decay_mask(params: Any) -> Any:
    """
    Return a mask tree matching the params structure.
    True  = apply weight decay to this parameter.
    False = exclude from weight decay.

    Excluded parameters:
      - 'scale' : RMSNorm / LayerNorm scale (γ)
      - 'bias'  : any bias term
      - 'pos_embedding' : learned positional embeddings

    Why exclude norm scale weights:
        Weight decay penalizes large magnitudes. Normalisation scale parameters
        are supposed to grow freely to control activation scale — decaying them
        distorts the learned scale and degrades training. Every frontier model
        (Llama, DeepSeek, GPT-4) excludes norm parameters from weight decay.

    FIX: was using optax.adamw(weight_decay=...) which applies decay to ALL
    parameters. Now uses masked decay via optax.add_decayed_weights.
    """
    # Flatten the nested param dict to a flat dict with tuple keys
    flat = traverse_util.flatten_dict(params)

    # The last element of the key tuple is the parameter name
    no_decay = {"scale", "bias", "pos_embedding"}
    mask_flat = {k: (k[-1] not in no_decay) for k in flat}

    # Unflatten back to the original nested structure
    return traverse_util.unflatten_dict(mask_flat)


# ------------------------------------------------------------
# Optimizer builders
# ------------------------------------------------------------

def build_adamw(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """
    AdamW with:
      - Gradient clipping (before parameter update)
      - Adam moment estimation
      - Masked weight decay (excludes norm params)
      - Learning rate schedule

    We build AdamW manually via optax.chain rather than using optax.adamw,
    because optax.adamw does not support masked weight decay out of the box.
    Building from primitives gives us full control.

    Parameter update rule:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t                  (first moment)
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²                 (second moment)
        m̂_t = m_t / (1 - β1^t)                               (bias correction)
        v̂_t = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

    β2=0.95 note:
        Standard Adam uses β2=0.999. All frontier models (Llama 3, DeepSeek V3,
        MiniCPM) use β2=0.95. Lower β2 makes the second moment adapt faster to
        gradient magnitude changes. With WSD schedule's large LR swings,
        faster adaptation = better stability.
    """

    return optax.chain(
        # 1. Clip gradient norm FIRST (before any scaling)
        optax.clip_by_global_norm(config.optimizer.gradient_clip),

        # 2. Adam moment estimation (no LR here — applied separately)
        optax.scale_by_adam(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
        ),

        # 3. Masked weight decay: only on non-norm, non-bias params
        #    add_decayed_weights adds λ * θ to the update
        optax.add_decayed_weights(
            weight_decay=config.optimizer.weight_decay,
            mask=get_weight_decay_mask,
        ),

        # 4. Apply learning rate schedule (negative: gradient descent)
        optax.scale_by_learning_rate(schedule),
    )


def build_adafactor(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """
    Adafactor: memory-efficient optimizer that factors the second moment.
    Uses much less memory than Adam at the cost of some stability.
    Good for very large models where Adam optimizer state doesn't fit.
    For 143M, Adam is preferred.

    FIX: was 'config.optmizee' (typo) — now 'config.optimizer'.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.gradient_clip),
        optax.adafactor(learning_rate=schedule),
    )


def build_lion(config: LaughLMConfig, schedule: Callable) -> optax.GradientTransformation:
    """
    Lion optimizer (EvoLved Sign Momentum).
    Uses only the sign of the gradient update — lower memory than Adam.
    Reported to match or beat Adam at scale, but less tested at 143M range.

    FIX: was missing return statement (returned None silently).
    """
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.gradient_clip),
        optax.lion(
            learning_rate=schedule,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
        ),
    )


# ------------------------------------------------------------
# Main factory
# ------------------------------------------------------------

def build_optimizer(
    config: LaughLMConfig,
    schedule: Callable,
) -> optax.GradientTransformation:
    """
    Build optimizer from config.

    Parameters
    ----------
    config   : LaughLMConfig
    schedule : callable step → lr, built by build_scheduler()

    The schedule must be passed in here (not built internally) because
    the same schedule object is also used for logging the current LR.

    FIX: previous version kept schedule and optimizer as separate objects,
    meaning the LR schedule was never applied to parameter updates.
    Now the schedule is baked into the optimizer via optax.chain.

    FIX: 'out_type' → 'opt_type' (NameError).
    FIX: build_optimizer now always returns — previous version returned None
         when gradient_clip was None.
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
