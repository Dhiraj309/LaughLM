import jax
import jax.numpy as jnp
import optax

from typing import Callable

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Device / step helpers
# ------------------------------------------------------------

def _resolve_num_devices(
    num_devices: int | None = None,
) -> int:
    """
    Resolve runtime device count.

    Scheduler construction may happen before trainer has a cached
    device count, so this helper falls back to jax.device_count().
    """

    if num_devices is None:
        num_devices = jax.device_count()

    num_devices = int(num_devices)

    if num_devices <= 0:
        raise ValueError(
            f"num_devices must be > 0, got {num_devices}"
        )

    return num_devices


def compute_tokens_per_step(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> int:
    """
    Compute tokens processed per optimizer update.

    Formula
    -------
    tokens_per_step =
        seq_len
        × micro_batch_per_device
        × num_devices
        × gradient_accumulation

    This is the effective token batch per optimizer step.
    """

    num_devices = _resolve_num_devices(
        num_devices
    )

    tokens_per_step = (
        int(config.runtime.seq_len)
        * int(config.runtime.micro_batch_per_device)
        * int(num_devices)
        * int(config.runtime.gradient_accumulation)
    )

    if tokens_per_step <= 0:
        raise ValueError(
            "Computed tokens_per_step <= 0.\n"
            "Check:\n"
            "  runtime.seq_len\n"
            "  runtime.micro_batch_per_device\n"
            "  runtime.gradient_accumulation\n"
            "  num_devices"
        )

    return int(tokens_per_step)


def _steps_from_token_budget(
    *,
    token_budget: int,
    tokens_per_step: int,
    label: str,
) -> int:
    """
    Convert a token budget into optimizer-update steps.
    """

    token_budget = int(token_budget)
    tokens_per_step = int(tokens_per_step)

    steps = token_budget // tokens_per_step

    if steps <= 0:
        raise ValueError(
            f"Computed {label}_steps <= 0.\n"
            f"  {label}_tokens:   {token_budget:,}\n"
            f"  tokens_per_step: {tokens_per_step:,}\n"
            "Increase token budget or reduce effective batch size."
        )

    return int(steps)


def get_scheduler_horizon_tokens(
    config: LaughLMConfig,
) -> int:
    """
    Return LR scheduler horizon in tokens.

    New behavior
    ------------
    If scheduler.horizon_tokens is set:
        use scheduler.horizon_tokens

    Backward-compatible behavior
    ----------------------------
    If scheduler.horizon_tokens is not set:
        use runtime.total_tokens

    This separates:
        runtime.total_tokens      = current stage stop target
        scheduler.horizon_tokens  = LR schedule horizon
    """

    horizon_tokens = getattr(
        config.scheduler,
        "horizon_tokens",
        None,
    )

    if horizon_tokens is None:
        return int(config.runtime.total_tokens)

    return int(horizon_tokens)


# ------------------------------------------------------------
# Runtime stop-step computation
# ------------------------------------------------------------

def compute_total_steps(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> int:
    """
    Compute runtime stop steps from runtime.total_tokens.

    Important
    ---------
    This is the TRAINING STOP TARGET.

    It is intentionally not always the LR scheduler horizon.

    For iterative pretraining:

        runtime.total_tokens:
            current cumulative stage target
            example: 1B, then 2B, then 5B

        scheduler.horizon_tokens:
            fixed LR horizon
            example: 20B or 40B

    This lets you extend training without reshaping the LR curve.
    """

    tokens_per_step = compute_tokens_per_step(
        config,
        num_devices,
    )

    total_steps = _steps_from_token_budget(
        token_budget=config.runtime.total_tokens,
        tokens_per_step=tokens_per_step,
        label="runtime",
    )

    print(
        f"[runtime] total_tokens: "
        f"{int(config.runtime.total_tokens):,}"
    )

    print(
        f"[runtime] tokens_per_step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"[runtime] total_steps: "
        f"{total_steps:,}"
    )

    return int(total_steps)


# ------------------------------------------------------------
# Scheduler horizon-step computation
# ------------------------------------------------------------

def compute_scheduler_total_steps(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> int:
    """
    Compute LR scheduler horizon steps.

    Uses scheduler.horizon_tokens when provided.
    Otherwise uses runtime.total_tokens for backward compatibility.
    """

    tokens_per_step = compute_tokens_per_step(
        config,
        num_devices,
    )

    horizon_tokens = get_scheduler_horizon_tokens(
        config
    )

    total_steps = _steps_from_token_budget(
        token_budget=horizon_tokens,
        tokens_per_step=tokens_per_step,
        label="scheduler",
    )

    print(
        f"[scheduler] runtime_total_tokens: "
        f"{int(config.runtime.total_tokens):,}"
    )

    print(
        f"[scheduler] horizon_tokens: "
        f"{horizon_tokens:,}"
    )

    print(
        f"[scheduler] tokens_per_step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"[scheduler] total_steps: "
        f"{total_steps:,}"
    )

    return int(total_steps)


# ------------------------------------------------------------
# Warmup helper
# ------------------------------------------------------------

def _resolve_warmup_steps(
    *,
    config: LaughLMConfig,
    total_steps: int,
    require_for_wsd: bool = False,
) -> int:
    """
    Resolve warmup steps from either warmup_steps or warmup_fraction.
    """

    if config.scheduler.warmup_steps is not None:
        warmup = int(
            config.scheduler.warmup_steps
        )

    elif config.scheduler.warmup_fraction is not None:
        warmup = int(
            total_steps
            * float(config.scheduler.warmup_fraction)
        )

    else:
        if require_for_wsd:
            raise ValueError(
                "WSD scheduler requires either "
                "scheduler.warmup_steps or scheduler.warmup_fraction."
            )

        warmup = max(
            int(total_steps * 0.01),
            1,
        )

    if warmup < 0:
        raise ValueError(
            f"warmup must be >= 0, got {warmup}"
        )

    return int(warmup)


def _validate_warmup_less_than_total(
    *,
    warmup: int,
    total_steps: int,
):
    if warmup >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup:,}) must be < "
            f"scheduler total_steps ({total_steps:,})"
        )


# ------------------------------------------------------------
# Cosine decay with warmup
# ------------------------------------------------------------

def build_cosine_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → cosine decay.

    Uses scheduler.horizon_tokens when provided.
    """

    lr = float(config.optimizer.learning_rate)
    min_ratio = float(config.scheduler.min_lr_ratio)

    total_steps = compute_scheduler_total_steps(
        config,
        num_devices,
    )

    warmup = _resolve_warmup_steps(
        config=config,
        total_steps=total_steps,
    )

    _validate_warmup_less_than_total(
        warmup=warmup,
        total_steps=total_steps,
    )

    print(
        f"[scheduler] cosine warmup: "
        f"{warmup:,}"
    )

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=max(warmup, 1),
        decay_steps=max(total_steps, 1),
        end_value=lr * min_ratio,
    )


# ------------------------------------------------------------
# Linear decay with warmup
# ------------------------------------------------------------

def build_linear_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → linear decay.

    Uses scheduler.horizon_tokens when provided.
    """

    lr = float(config.optimizer.learning_rate)
    min_ratio = float(config.scheduler.min_lr_ratio)

    total_steps = compute_scheduler_total_steps(
        config,
        num_devices,
    )

    warmup = _resolve_warmup_steps(
        config=config,
        total_steps=total_steps,
    )

    _validate_warmup_less_than_total(
        warmup=warmup,
        total_steps=total_steps,
    )

    min_lr = lr * min_ratio

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=min_lr,
        transition_steps=max(total_steps - warmup, 1),
    )

    print(
        "[scheduler] linear phases:\n"
        f"  warmup: {warmup:,}\n"
        f"  decay:  {max(total_steps - warmup, 1):,}"
    )

    return optax.join_schedules(
        schedules=[
            warmup_sched,
            decay_sched,
        ],
        boundaries=[
            warmup,
        ],
    )


# ------------------------------------------------------------
# Inverse-square-root scheduler
# ------------------------------------------------------------

def build_rsqrt_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → inverse sqrt decay.

    This schedule is naturally open-ended, but when warmup_fraction is used
    it resolves against scheduler.horizon_tokens for consistency.
    """

    lr = float(config.optimizer.learning_rate)

    if config.scheduler.warmup_steps is not None:
        warmup = int(
            config.scheduler.warmup_steps
        )

    elif config.scheduler.warmup_fraction is not None:
        total_steps = compute_scheduler_total_steps(
            config,
            num_devices,
        )

        warmup = int(
            total_steps
            * float(config.scheduler.warmup_fraction)
        )

    else:
        raise ValueError(
            "rsqrt scheduler requires scheduler.warmup_steps "
            "or scheduler.warmup_fraction."
        )

    if warmup <= 0:
        raise ValueError(
            "rsqrt scheduler requires warmup > 0"
        )

    warmup_f = jnp.asarray(
        warmup,
        dtype=jnp.float32,
    )

    print(
        f"[scheduler] rsqrt warmup: "
        f"{warmup:,}"
    )

    def schedule(step):
        step_f = jnp.asarray(
            step,
            dtype=jnp.float32,
        )

        step_f = jnp.maximum(
            step_f,
            1.0,
        )

        scale = jnp.minimum(
            step_f ** -0.5,
            step_f * warmup_f ** -1.5,
        )

        return lr * scale

    return schedule


# ------------------------------------------------------------
# WSD scheduler
# ------------------------------------------------------------

def build_wsd_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → Stable → Decay.

    Uses scheduler.horizon_tokens when provided.

    This is the recommended scheduler for staged pretraining:

        runtime.total_tokens changes stage-by-stage.
        scheduler.horizon_tokens stays fixed.

    This prevents LR restarts / LR curve reshaping on resume.
    """

    lr = float(config.optimizer.learning_rate)
    min_ratio = float(config.scheduler.min_lr_ratio)

    total_steps = compute_scheduler_total_steps(
        config,
        num_devices,
    )

    warmup = _resolve_warmup_steps(
        config=config,
        total_steps=total_steps,
        require_for_wsd=True,
    )

    stable_fraction = float(
        config.scheduler.stable_fraction
    )

    stable_steps = int(
        total_steps
        * stable_fraction
    )

    if config.scheduler.decay_steps is not None:
        decay_steps = int(
            config.scheduler.decay_steps
        )

        stable_steps = (
            total_steps
            - warmup
            - decay_steps
        )

    else:
        decay_steps = (
            total_steps
            - warmup
            - stable_steps
        )

    if warmup < 0:
        raise ValueError(
            f"warmup must be >= 0, got {warmup}"
        )

    if stable_steps < 0:
        raise ValueError(
            f"Computed stable_steps is negative ({stable_steps}). "
            "Check scheduler.stable_fraction or scheduler.decay_steps."
        )

    if decay_steps < 0:
        raise ValueError(
            f"Computed decay_steps is negative ({decay_steps}). "
            "Reduce warmup or stable_fraction."
        )

    if warmup + stable_steps >= total_steps:
        raise ValueError(
            "Invalid WSD schedule:\n"
            f"  warmup: {warmup:,}\n"
            f"  stable: {stable_steps:,}\n"
            f"  total:  {total_steps:,}\n"
            "No room for decay. Reduce warmup/stable_fraction or "
            "increase scheduler.horizon_tokens."
        )

    min_lr = lr * min_ratio

    print(
        "[scheduler] WSD phases:\n"
        f"  warmup: {warmup:,}\n"
        f"  stable: {stable_steps:,}\n"
        f"  decay:  {decay_steps:,}"
    )

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    stable_sched = optax.constant_schedule(
        lr
    )

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=min_lr,
        transition_steps=max(decay_steps, 1),
    )

    return optax.join_schedules(
        schedules=[
            warmup_sched,
            stable_sched,
            decay_sched,
        ],
        boundaries=[
            warmup,
            warmup + stable_steps,
        ],
    )


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------

def build_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Build learning-rate scheduler.

    Returns
    -------
    Callable
        step -> learning_rate
    """

    sched_type = config.scheduler.type

    print(
        f"[scheduler] type: {sched_type}"
    )

    if sched_type == "cosine":
        return build_cosine_scheduler(
            config,
            num_devices,
        )

    if sched_type == "linear":
        return build_linear_scheduler(
            config,
            num_devices,
        )

    if sched_type == "rsqrt":
        return build_rsqrt_scheduler(
            config,
            num_devices,
        )

    if sched_type == "wsd":
        return build_wsd_scheduler(
            config,
            num_devices,
        )

    raise ValueError(
        f"Unknown scheduler type: '{sched_type}'. "
        "Valid options: cosine, linear, rsqrt, wsd."
    )
