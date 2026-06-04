from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def validate_config(config: LaughLMConfig) -> None:
    """
    Run all cross-field validation rules.

    Raises
    ------
    ValueError
        If any configuration rule is violated.
    """

    _validate_runtime_backend(config)
    _validate_attention_heads(config)
    _validate_gqa_kv_heads(config)
    _validate_positional(config)
    _validate_norm_residual_compatibility(config)
    _validate_moe_requirements(config)
    _validate_scheduler_horizon(config)
    _validate_wsd_scheduler(config)

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------

def _tokens_per_step(config: LaughLMConfig) -> int:
    """
    Compute optimizer-update token count.

    Formula
    -------
    tokens_per_step =
        seq_len
        × micro_batch_per_device
        × data_parallel
        × gradient_accumulation

    Notes
    -----
    Validation uses config.parallelism.data_parallel because this file
    runs before trainer.py has runtime JAX device count context.
    trainer/scheduler.py should use real num_devices at runtime.
    """

    return int(
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * config.parallelism.data_parallel
        * config.runtime.gradient_accumulation
    )


def _scheduler_horizon_tokens(config: LaughLMConfig) -> int:
    """
    LR scheduler horizon in tokens.

    Backward compatible behavior:
    - If scheduler.horizon_tokens is unset, use runtime.total_tokens.

    New staged-training behavior:
    - runtime.total_tokens = current stage stop target
    - scheduler.horizon_tokens = fixed LR schedule horizon
    """

    horizon = getattr(
        config.scheduler,
        "horizon_tokens",
        None,
    )

    if horizon is None:
        return int(config.runtime.total_tokens)

    return int(horizon)


def _scheduler_total_steps(config: LaughLMConfig) -> int:
    tokens_per_step = _tokens_per_step(config)

    if tokens_per_step <= 0:
        raise ValueError(
            "Computed tokens_per_step <= 0. "
            "Check seq_len, micro_batch_per_device, "
            "parallelism.data_parallel, and gradient_accumulation."
        )

    horizon_tokens = _scheduler_horizon_tokens(config)

    return int(
        horizon_tokens // tokens_per_step
    )


# ------------------------------------------------------------
# Validation Rules
# ------------------------------------------------------------

def _validate_attention_heads(config: LaughLMConfig) -> None:
    """
    Ensure head dimension divides model dimension.
    """

    d_model = config.model.d_model
    num_heads = config.model.num_heads

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"head_dim = {d_model} / {num_heads} = {d_model / num_heads:.2f} "
            f"(not integer)."
        )


def _validate_gqa_kv_heads(config: LaughLMConfig) -> None:
    """
    When GQA is selected, num_kv_heads must:
      - Be specified
      - Be <= num_heads
      - Divide num_heads evenly
    """

    if config.architecture.attention_variant != "gqa":
        return

    num_heads = config.model.num_heads
    num_kv_heads = config.model.num_kv_heads

    if num_kv_heads is None:
        raise ValueError(
            "attention_variant='gqa' requires model.num_kv_heads to be set. "
            f"For a {num_heads}-head model, typical values are "
            f"{num_heads} for MHA-compatible GQA, "
            f"{max(num_heads // 4, 1)} for 4:1 GQA, or "
            f"{max(num_heads // 8, 1)} for 8:1 GQA."
        )

    if num_kv_heads <= 0:
        raise ValueError(
            f"num_kv_heads must be > 0, got {num_kv_heads}."
        )

    if num_kv_heads > num_heads:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must be <= num_heads ({num_heads})."
        )

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) so each KV group covers "
            f"the same number of Q heads."
        )


def _validate_positional(config: LaughLMConfig) -> None:
    """
    Validate positional embedding compatibility.
    """

    positional = config.architecture.positional

    if positional == "alibi":
        raise ValueError(
            "positional='alibi' is not yet implemented. "
            "Use 'rope' for current TPU PMAP training."
        )


def _validate_norm_residual_compatibility(config: LaughLMConfig) -> None:
    """
    DeepNorm requires matching residual configuration.
    """

    norm = config.architecture.normalization
    residual = config.architecture.residual

    if norm == "deep_norm" and residual != "deep_norm":
        raise ValueError(
            "normalization='deep_norm' requires residual='deep_norm'. "
            "DeepNorm uses coordinated alpha/beta scaling between norm "
            "and residual."
        )


def _validate_moe_requirements(config: LaughLMConfig) -> None:
    """
    Placeholder validation for MoE architecture.
    """

    if config.architecture.ffn_type == "moe":
        raise ValueError(
            "ffn_type='moe' selected but MoE is not yet implemented. "
            "Use 'swiglu' for the current PMAP path."
        )


def _validate_scheduler_horizon(config: LaughLMConfig) -> None:
    """
    Validate staged-training scheduler semantics.

    runtime.total_tokens:
        Current cumulative stop target for this stage.

    scheduler.horizon_tokens:
        Fixed LR schedule horizon.

    For iterative pretraining, horizon_tokens should stay fixed while
    runtime.total_tokens increases.
    """

    tokens_per_step = _tokens_per_step(config)

    if tokens_per_step <= 0:
        raise ValueError(
            "Computed tokens_per_step <= 0. "
            "Check seq_len, micro_batch_per_device, "
            "parallelism.data_parallel, and gradient_accumulation."
        )

    runtime_steps = config.runtime.total_tokens // tokens_per_step

    if runtime_steps <= 0:
        raise ValueError(
            "runtime.total_tokens produces zero optimizer steps.\n"
            f"  runtime.total_tokens: {config.runtime.total_tokens:,}\n"
            f"  tokens_per_step:      {tokens_per_step:,}\n"
            "Increase runtime.total_tokens or reduce effective batch size."
        )

    horizon = getattr(
        config.scheduler,
        "horizon_tokens",
        None,
    )

    if horizon is None:
        return

    if horizon < config.runtime.total_tokens:
        raise ValueError(
            "scheduler.horizon_tokens must be >= runtime.total_tokens.\n"
            f"  scheduler.horizon_tokens: {horizon:,}\n"
            f"  runtime.total_tokens:     {config.runtime.total_tokens:,}\n"
            "For staged training, runtime.total_tokens is the current "
            "stop target, while scheduler.horizon_tokens is the fixed "
            "LR schedule horizon."
        )

    scheduler_steps = horizon // tokens_per_step

    if scheduler_steps <= 0:
        raise ValueError(
            "scheduler.horizon_tokens produces zero scheduler steps.\n"
            f"  scheduler.horizon_tokens: {horizon:,}\n"
            f"  tokens_per_step:          {tokens_per_step:,}\n"
            "Increase scheduler.horizon_tokens or reduce effective batch size."
        )


def _validate_wsd_scheduler(config: LaughLMConfig) -> None:
    """
    WSD-specific cross-field validation.

    Uses scheduler.horizon_tokens when provided.

    This prevents LR schedule reshaping when runtime.total_tokens is
    extended stage-by-stage.
    """

    if config.scheduler.type != "wsd":
        return

    stable_fraction = config.scheduler.stable_fraction

    if not (0.0 < stable_fraction < 1.0):
        raise ValueError(
            f"scheduler.stable_fraction must be in (0, 1), got "
            f"{stable_fraction}."
        )

    total_steps = _scheduler_total_steps(config)

    if total_steps <= 0:
        raise ValueError(
            f"Invalid scheduler total_steps computed: {total_steps}. "
            "Check scheduler.horizon_tokens/runtime.total_tokens and "
            "batch configuration."
        )

    if config.scheduler.warmup_steps is not None:
        warmup = int(config.scheduler.warmup_steps)

    elif config.scheduler.warmup_fraction is not None:
        warmup = int(
            total_steps
            * config.scheduler.warmup_fraction
        )

    else:
        raise ValueError(
            "WSD scheduler requires either scheduler.warmup_steps "
            "or scheduler.warmup_fraction."
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
            f"warmup must be >= 0, got {warmup}."
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
            f"  scheduler_horizon_tokens: {_scheduler_horizon_tokens(config):,}\n"
            f"  tokens_per_step:          {_tokens_per_step(config):,}\n"
            f"  total_steps:              {total_steps:,}\n"
            f"  warmup:                   {warmup:,}\n"
            f"  stable:                   {stable_steps:,}\n"
            f"  decay:                    {decay_steps:,}\n"
            "No room left for decay phase. "
            "Reduce warmup, reduce stable_fraction, or increase horizon_tokens."
        )

def _validate_runtime_backend(config: LaughLMConfig) -> None:
    """
    Validate runtime backend naming.

    Accepted canonical backends:
      - pmap
      - fsdp
      - parallel3d
      - moe

    Temporary compatibility alias:
      - gspmd -> fsdp

    This validation only checks naming. Trainer availability is checked
    by entrypoints/scripts so config loading remains useful for future
    reserved backend configs.
    """

    backend = str(config.runtime.backend)

    canonical_backend = getattr(
        config.runtime,
        "canonical_backend",
        backend,
    )

    valid_canonical = {
        "pmap",
        "fsdp",
        "parallel3d",
        "moe",
    }

    if canonical_backend not in valid_canonical:
        raise ValueError(
            "Invalid runtime.backend.\n"
            f"  backend:           {backend!r}\n"
            f"  canonical_backend: {canonical_backend!r}\n"
            f"  valid:             {sorted(valid_canonical)}\n"
            "Temporary alias supported: 'gspmd' -> 'fsdp'."
        )
