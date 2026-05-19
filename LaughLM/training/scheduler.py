import jax
import optax
from typing import Callable

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Utility: compute total training steps
# ------------------------------------------------------------

def compute_total_steps(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> int:
    """
    Derive total optimizer update steps from cumulative token budget.

    Formula
    -------
    tokens_per_step =
        seq_len
        × micro_batch_per_device
        × num_devices
        × gradient_accumulation

    total_steps =
        total_tokens // tokens_per_step

    Resume Safety
    -------------
    total_tokens is interpreted as the FULL cumulative
    training horizon, NOT "additional tokens".

    Example:
        Run 1:
            total_tokens = 100M

        Resume:
            total_tokens = 200M

    The scheduler then smoothly extends training to the
    200M-token horizon without restarting LR schedules.

    Parameters
    ----------
    config : LaughLMConfig

    num_devices : int, optional
        Real runtime device count.
        If None, uses jax.device_count().

    Returns
    -------
    int
        Total optimizer steps.
    """

    if num_devices is None:
        num_devices = jax.device_count()

    if num_devices <= 0:
        raise ValueError(
            f"num_devices must be > 0, got {num_devices}"
        )

    tokens_per_step = (
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * num_devices
        * config.runtime.gradient_accumulation
    )

    if tokens_per_step <= 0:
        raise ValueError(
            "Computed tokens_per_step <= 0.\n"
            "Check:\n"
            "  seq_len\n"
            "  micro_batch_per_device\n"
            "  gradient_accumulation\n"
            "  num_devices"
        )

    total_steps = (
        config.runtime.total_tokens
        // tokens_per_step
    )

    if total_steps <= 0:
        raise ValueError(
            "Computed total_steps <= 0.\n"
            "Increase runtime.total_tokens or "
            "reduce effective batch size."
        )

    print(
        f"[scheduler] total_tokens: "
        f"{config.runtime.total_tokens:,}"
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
# Cosine decay with warmup
# ------------------------------------------------------------

def build_cosine_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → cosine decay.

    Good for:
    - fine-tuning
    - short pretraining
    """

    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio

    total_steps = compute_total_steps(
        config,
        num_devices,
    )

    # --------------------------------------------------------
    # Warmup resolution
    # --------------------------------------------------------

    if config.scheduler.warmup_steps is not None:

        warmup = config.scheduler.warmup_steps

    elif (
        config.scheduler.warmup_fraction
        is not None
    ):

        warmup = int(
            total_steps
            * config.scheduler.warmup_fraction
        )

    else:

        # safe default
        warmup = max(
            int(total_steps * 0.01),
            1,
        )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if warmup >= total_steps:

        raise ValueError(
            f"warmup_steps ({warmup}) "
            f"must be < total_steps "
            f"({total_steps})"
        )

    print(
        f"[scheduler] cosine warmup: "
        f"{warmup:,}"
    )

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=lr * min_ratio,
    )


# ------------------------------------------------------------
# Linear decay
# ------------------------------------------------------------

def build_linear_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → linear decay.
    """

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate

    total_steps = compute_total_steps(
        config,
        num_devices,
    )

    if warmup >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup}) must be "
            f"< total_steps ({total_steps})"
        )

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=0.0,
        transition_steps=max(total_steps - warmup, 1),
    )

    return optax.join_schedules(
        schedules=[
            warmup_sched,
            decay_sched,
        ],
        boundaries=[warmup],
    )


# ------------------------------------------------------------
# Inverse square root
# ------------------------------------------------------------

def build_rsqrt_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → inverse sqrt decay.

    Used in:
    - Transformer paper
    - T5

    Step-based schedule.
    """

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate

    if warmup <= 0:
        raise ValueError(
            "rsqrt scheduler requires "
            "warmup_steps > 0"
        )

    def schedule(step: int) -> float:

        step = max(step, 1)

        scale = min(
            step ** -0.5,
            step * warmup ** -1.5,
        )

        return lr * scale

    return schedule


# ------------------------------------------------------------
# WSD Scheduler
# ------------------------------------------------------------

def build_wsd_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Warmup → Stable → Decay

    Reference:
    MiniCPM (2024)

    Resume-friendly schedule design.
    """

    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio

    total_steps = compute_total_steps(
        config,
        num_devices,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if config.scheduler.warmup_steps is not None:

        warmup = config.scheduler.warmup_steps

    elif getattr(
        config.scheduler,
        "warmup_fraction",
        None,
    ) is not None:

        warmup = int(
            total_steps
            * config.scheduler.warmup_fraction
        )

    else:
        raise ValueError(
            "WSD scheduler requires either "
            "'warmup_steps' or 'warmup_fraction'."
        )

    # --------------------------------------------------------
    # Stable phase
    # --------------------------------------------------------

    stable_fraction = config.scheduler.stable_fraction

    stable_steps = int(
        total_steps * stable_fraction
    )

    # --------------------------------------------------------
    # Decay phase
    # --------------------------------------------------------

    if config.scheduler.decay_steps is not None:

        decay_steps = config.scheduler.decay_steps

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

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if warmup < 0:
        raise ValueError(
            f"warmup must be >= 0, got {warmup}"
        )

    if stable_steps < 0:
        raise ValueError(
            f"Computed stable_steps is negative "
            f"({stable_steps}). "
            f"Check stable_fraction or decay_steps."
        )

    if decay_steps < 0:
        raise ValueError(
            f"Computed decay_steps is negative "
            f"({decay_steps}). "
            f"Reduce warmup or stable_fraction."
        )

    if warmup + stable_steps >= total_steps:
        raise ValueError(
            f"Invalid schedule:\n"
            f"  warmup: {warmup}\n"
            f"  stable: {stable_steps}\n"
            f"  total:  {total_steps}\n"
            f"→ No room for decay."
        )

    min_lr = lr * min_ratio

    print(
        "[scheduler] WSD phases:\n"
        f"  warmup: {warmup:,}\n"
        f"  stable: {stable_steps:,}\n"
        f"  decay:  {decay_steps:,}"
    )

    # --------------------------------------------------------
    # Phase 1: Warmup
    # --------------------------------------------------------

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=max(warmup, 1),
    )

    # --------------------------------------------------------
    # Phase 2: Stable
    # --------------------------------------------------------

    stable_sched = optax.constant_schedule(lr)

    # --------------------------------------------------------
    # Phase 3: Decay
    # --------------------------------------------------------

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=min_lr,
        transition_steps=max(decay_steps, 1),
    )

    # --------------------------------------------------------
    # Compose
    # --------------------------------------------------------

    schedule = optax.join_schedules(
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

    return schedule


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------

def build_scheduler(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Callable:
    """
    Build learning rate scheduler.

    Returns
    -------
    Callable:
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
        f"Valid options: cosine, linear, rsqrt, wsd."
    )
