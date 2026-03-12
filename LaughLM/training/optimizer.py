
import jax
import optax
from LaughLM.config.schema import LaughLMConfig
from LaughLM.training.scheduler import build_scheduler


# ─────────────────────────────────────────────────────────────
# Weight Decay Mask
# ─────────────────────────────────────────────────────────────

def _weight_decay_mask(params):
    """
    Returns a pytree of booleans mirroring params.
    True  = apply weight decay to this parameter.
    False = skip weight decay (norm scales, biases, embeddings).

    Why: optax.add_decayed_weights applies decay to everything by default,
    including RMSNorm/LayerNorm scale params. Decaying those distorts learned
    normalization and degrades training. All frontier models exclude them.
    """

    def should_decay(path, _):
        # path is a tuple of keys, e.g. ('layers_0', 'attn', 'scale')
        leaf = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        return leaf not in ("scale", "bias", "embedding")

    return jax.tree_util.tree_map_with_path(should_decay, params)


# ─────────────────────────────────────────────────────────────
# Per-optimizer builders (internal helpers)
# ─────────────────────────────────────────────────────────────

def _build_adamw_chain(config: LaughLMConfig, schedule):
    """
    AdamW as an explicit transformation chain with schedule baked in.

    Chain order matters:
      1. clip_by_global_norm  — clip raw gradients first
      2. scale_by_adam        — compute moment estimates (no LR yet)
      3. add_decayed_weights  — decoupled weight decay (masked)
      4. scale_by_learning_rate(schedule) — apply LR from WSD/cosine curve

    scale_by_learning_rate applies a negative sign automatically,
    so updates point in the descent direction without manual negation.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.gradient_clip),
        optax.scale_by_adam(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
        ),
        optax.add_decayed_weights(
            weight_decay=config.optimizer.weight_decay,
            mask=_weight_decay_mask,
        ),
        optax.scale_by_learning_rate(schedule),
    )


def _build_adafactor_chain(config: LaughLMConfig, schedule):
    """
    Adafactor with schedule. Adafactor uses factored second-moment
    estimation — much lower memory than Adam (no full v vector).
    Good alternative when optimizer memory is tight on TPU.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.gradient_clip),
        optax.scale_by_factored_rms(),
        optax.scale_by_learning_rate(schedule),
    )


def _build_lion_chain(config: LaughLMConfig, schedule):
    """
    Lion optimizer. Uses sign of gradient — memory efficient,
    often matches AdamW at lower LR (typical: 3-10x lower than AdamW LR).
    """
    return optax.chain(
        optax.clip_by_global_norm(config.optimizer.gradient_clip),
        optax.scale_by_lion(
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
        ),
        optax.add_decayed_weights(
            weight_decay=config.optimizer.weight_decay,
            mask=_weight_decay_mask,
        ),
        optax.scale_by_learning_rate(schedule),
    )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def build_optimizer(config: LaughLMConfig):
    """
    Build optimizer with learning rate schedule baked in.

    The schedule is NOT a separate object — it lives inside the optimizer
    chain via scale_by_learning_rate(schedule). Optax calls schedule(step)
    automatically on every optimizer.update() call using the step counter
    stored in opt_state. You never pass the step manually.

    Usage
    -----
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(params)

    # In training loop:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    """

    # Build the schedule first — it becomes part of the optimizer chain
    schedule = build_scheduler(config)

    opt_type = config.optimizer.type

    if opt_type == "adamw":
        return _build_adamw_chain(config, schedule)

    elif opt_type == "adafactor":
        return _build_adafactor_chain(config, schedule)

    elif opt_type == "lion":
        return _build_lion_chain(config, schedule)

    elif opt_type == "muon":
        raise NotImplementedError("Muon optimizer not yet implemented")

    else:
        raise ValueError(f"Unknown optimizer type: {opt_type!r}")
