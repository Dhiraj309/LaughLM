import optax
from typing import Callable

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Utility: compute total training steps
# ------------------------------------------------------------

def compute_total_steps(config: LaughLMConfig) -> int:
    """
    Derive total gradient update steps from config.

    tokens_per_step = seq_len × micro_batch × data_parallel × grad_accumulation
    total_steps     = total_tokens / tokens_per_step
    """
    tokens_per_step = (
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * config.parallelism.data_parallel
        * config.runtime.gradient_accumulation
    )
    return config.runtime.total_tokens // tokens_per_step


# ------------------------------------------------------------
# Cosine decay with warmup
# ------------------------------------------------------------

def build_cosine_scheduler(config: LaughLMConfig) -> Callable:
    """
    Warmup → cosine decay.

    Standard schedule for fine-tuning or short pretraining runs.
    NOT recommended for production pretraining — use WSD instead.
    WSD allows training extension without restart; cosine does not.
    """
    warmup      = config.scheduler.warmup_steps
    lr          = config.optimizer.learning_rate
    min_ratio   = config.scheduler.min_lr_ratio
    total_steps = compute_total_steps(config)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=lr * min_ratio,
    )


# ------------------------------------------------------------
# Linear (warmup → linear decay)
# ------------------------------------------------------------

def build_linear_scheduler(config: LaughLMConfig) -> Callable:
    """
    Linear warmup → linear decay to zero.
    """
    warmup      = config.scheduler.warmup_steps
    lr          = config.optimizer.learning_rate
    total_steps = compute_total_steps(config)

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup,
    )

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=0.0,
        transition_steps=total_steps - warmup,
    )

    return optax.join_schedules(
        schedules=[warmup_sched, decay_sched],
        boundaries=[warmup],
    )


# ------------------------------------------------------------
# Inverse square root (T5 / original Transformer)
# ------------------------------------------------------------

def build_rsqrt_scheduler(config: LaughLMConfig) -> Callable:
    """
    Warmup then inverse square root decay.
    LR ∝ 1/√step after warmup.
    Used in T5 and the original Transformer paper.
    """
    warmup = config.scheduler.warmup_steps
    lr     = config.optimizer.learning_rate

    def schedule(step: int) -> float:
        step = max(step, 1)
        scale = min(
            step ** -0.5,
            step * warmup ** -1.5,
        )
        return lr * scale

    return schedule


# ------------------------------------------------------------
# WSD — Warmup → Stable → Decay
# ------------------------------------------------------------

def build_wsd_scheduler(config: LaughLMConfig) -> Callable:
    """
    Warmup-Stable-Decay learning rate schedule (MiniCPM, 2024).

    Three phases
    ------------
    1. Warmup  : linear 0 → max_lr
                 Duration: config.scheduler.warmup_steps
                 Purpose: prevent instability from random-initialized gradients.

    2. Stable  : constant max_lr
                 Duration: stable_fraction × total_steps (default 88%)
                 Purpose: bulk of learning. This phase is the "main branch" —
                 it can be extended indefinitely by loading this checkpoint
                 and continuing with the same schedule. No restart needed.
                 WSD's key operational advantage over cosine.

    3. Decay   : linear max_lr → min_lr (D2Z: 0.0)
                 Duration: total_steps - warmup - stable
                 Purpose: rapid convergence to sharp minimum.
                 D2Z (decay-to-zero) is proven better than decaying to 10%:
                 any non-zero final LR leaves energy in weights that
                 prevents full convergence. min_lr_ratio defaults to 0.0.

    Resuming training
    -----------------
    To extend training beyond the original total_steps:
      1. Save checkpoint at end of STABLE phase (before decay begins).
      2. Re-run with a new config: warmup_steps=0, larger total_tokens.
      3. The resumed run enters stable phase immediately (no re-warmup).
      4. Apply a new short decay at the new end.
    Do NOT resume from a post-decay checkpoint — the optimizer state has
    already converged and will not benefit from more stable-phase training.

    FIX: was hardcoding stable_steps = 60% of total, ignoring config fields.
         MiniCPM research shows optimal split is ~88% stable, ~10% decay, ~2% warmup.
         Now reads stable_fraction from SchedulerConfig (default 0.88).

    FIX: min_lr_ratio now defaults to 0.0 (D2Z).
         Previous config had 0.1 which MiniCPM ablations showed is suboptimal.
    """

    warmup      = config.scheduler.warmup_steps
    lr          = config.optimizer.learning_rate
    min_ratio   = config.scheduler.min_lr_ratio         # 0.0 for D2Z
    total_steps = compute_total_steps(config)

    # Compute stable phase length from fraction
    stable_fraction = config.scheduler.stable_fraction  # default 0.88
    stable_steps    = int(total_steps * stable_fraction)

    # Honor override if provided
    if config.scheduler.decay_steps is not None:
        decay_steps = config.scheduler.decay_steps
        # Recompute stable to fit: stable = total - warmup - decay
        stable_steps = total_steps - warmup - decay_steps
    else:
        decay_steps = total_steps - warmup - stable_steps

    min_lr = lr * min_ratio

    # Phase 1: Warmup
    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    # Phase 2: Stable (constant)
    stable_sched = optax.constant_schedule(lr)

    # Phase 3: Decay to min_lr (D2Z when min_lr=0.0)
    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=min_lr,
        transition_steps=max(decay_steps, 1),
    )

    schedule = optax.join_schedules(
        schedules=[warmup_sched, stable_sched, decay_sched],
        boundaries=[warmup, warmup + stable_steps],
    )

    return schedule


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_scheduler(config: LaughLMConfig) -> Callable:
    """
    Build learning rate schedule from config.

    Returns a callable: step (int) → learning_rate (float).
    This callable is passed to build_optimizer() to be baked into
    the optimizer chain via optax.scale_by_learning_rate.
    """

    sched_type = config.scheduler.type

    if sched_type == "cosine":
        return build_cosine_scheduler(config)

    if sched_type == "linear":
        return build_linear_scheduler(config)

    if sched_type == "rsqrt":
        return build_rsqrt_scheduler(config)

    if sched_type == "wsd":
        return build_wsd_scheduler(config)

    raise ValueError(
        f"Unknown scheduler type: '{sched_type}'. "
        f"Valid options: cosine, linear, rsqrt, wsd."
    )
