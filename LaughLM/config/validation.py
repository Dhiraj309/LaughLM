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
    _validate_dtype_alignment(config)
    _validate_parallelism_mesh_alignment(config)
    _validate_attention_mesh_compatibility(config)
    _validate_attention_heads(config)
    _validate_active_llama_architecture(config)
    _validate_attention_variant(config)
    _validate_gqa_kv_heads(config)
    _validate_scheduler_horizon(config)
    _validate_wsd_scheduler(config)
    _validate_optimizations(config)

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------

def _validate_dtype_alignment(config: LaughLMConfig) -> None:
    """Ensure legacy dtype fields cannot diverge from canonical SPMD policy."""
    canonical = config.spmd.dtype
    legacy = config.parallelism

    mismatches = []
    if canonical.param_dtype != legacy.param_dtype:
        mismatches.append(
            f"param_dtype: spmd={canonical.param_dtype!r}, "
            f"parallelism={legacy.param_dtype!r}"
        )
    if canonical.compute_dtype != legacy.compute_dtype:
        mismatches.append(
            f"compute_dtype: spmd={canonical.compute_dtype!r}, "
            f"parallelism={legacy.compute_dtype!r}"
        )

    if mismatches:
        raise ValueError(
            "Canonical spmd.dtype and legacy parallelism dtype fields must "
            "match during the dtype migration:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )

def _validate_optimizations(config: LaughLMConfig) -> None:
    """Validate optimization options."""
    opts = getattr(config, "optimizations", None)
    if opts is None:
        return

    valid_kernels = {"native", "tokamax"}
    if opts.kernel_backend not in valid_kernels:
        raise ValueError(
            f"Invalid optimizations.kernel_backend: {opts.kernel_backend!r}. "
            f"Expected one of {sorted(valid_kernels)}"
        )

    valid_data = {"native", "grain"}
    if opts.data_backend not in valid_data:
        raise ValueError(
            f"Invalid optimizations.data_backend: {opts.data_backend!r}. "
            f"Expected one of {sorted(valid_data)}"
        )

    valid_sharding = {"pmap", "fsdp", "maxtext_3d"}
    if opts.sharding_strategy not in valid_sharding:
        raise ValueError(
            f"Invalid optimizations.sharding_strategy: {opts.sharding_strategy!r}. "
            f"Expected one of {sorted(valid_sharding)}"
        )


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


def _validate_parallelism_mesh_alignment(config: LaughLMConfig) -> None:
    """
    Ensure legacy parallelism fields agree with the SPMD mesh.

    Why this matters:
    - scheduler validation uses parallelism.data_parallel
    - PMAP/FSDP trainers use runtime device/mesh counts
    - mismatch silently corrupts tokens_per_step, LR horizon,
      checkpoint metadata, and throughput reporting

    Current supported canonical backends:
    - pmap: pure data parallel
    - fsdp: data x fsdp, no tensor/sequence/pipeline yet

    parallel3d/moe are reserved and validated later by their trainers.
    """

    backend = str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )

    if backend not in {
        "pmap",
        "fsdp",
    }:
        return

    axis_sizes = config.spmd.mesh.axis_sizes()

    data_axis = int(
        axis_sizes.get(
            "data",
            1,
        )
    )

    fsdp_axis = int(
        axis_sizes.get(
            "fsdp",
            1,
        )
    )

    tensor_axis = int(
        axis_sizes.get(
            "tensor",
            1,
        )
    )

    sequence_axis = int(
        axis_sizes.get(
            "sequence",
            1,
        )
    )

    pipeline_axis = int(
        axis_sizes.get(
            "pipeline",
            1,
        )
    )

    mesh_total = int(
        config.spmd.mesh.total_devices()
    )

    legacy_data = int(
        config.parallelism.data_parallel
    )

    legacy_model = int(
        config.parallelism.model_parallel
    )

    if legacy_data != data_axis:
        raise ValueError(
            "parallelism.data_parallel must match spmd.mesh data axis.\n"
            f"  runtime.backend:             {config.runtime.backend!r}\n"
            f"  canonical backend:           {backend!r}\n"
            f"  parallelism.data_parallel:   {legacy_data}\n"
            f"  spmd.mesh axis_sizes[data]:  {data_axis}\n"
            "This field is still used by config-time scheduler validation."
        )

    if backend == "pmap":
        non_data_axes = {
            "fsdp": fsdp_axis,
            "tensor": tensor_axis,
            "sequence": sequence_axis,
            "pipeline": pipeline_axis,
        }

        active_non_data_axes = {
            name: size
            for name, size in non_data_axes.items()
            if size > 1
        }

        if active_non_data_axes:
            raise ValueError(
                "runtime.backend='pmap' requires pure data-parallel mesh.\n"
                f"  active non-data axes: {active_non_data_axes}\n"
                "Use runtime.backend='fsdp' for fsdp>1, or a future "
                "parallel3d backend for tensor/sequence axes."
            )

    expected_model_parallel = (
        fsdp_axis
        * tensor_axis
    )

    if legacy_model != expected_model_parallel:
        raise ValueError(
            "parallelism.model_parallel must match fsdp*tensor mesh axes "
            "for current PMAP/FSDP backends.\n"
            f"  runtime.backend:               {config.runtime.backend!r}\n"
            f"  canonical backend:             {backend!r}\n"
            f"  parallelism.model_parallel:    {legacy_model}\n"
            f"  spmd.mesh axis_sizes[fsdp]:    {fsdp_axis}\n"
            f"  spmd.mesh axis_sizes[tensor]:  {tensor_axis}\n"
            f"  expected model_parallel:       {expected_model_parallel}"
        )

    if legacy_data * legacy_model != mesh_total:
        raise ValueError(
            "Legacy parallelism product must match total SPMD mesh devices.\n"
            f"  data_parallel * model_parallel: {legacy_data * legacy_model}\n"
            f"  spmd.mesh.total_devices():      {mesh_total}"
        )

    if backend == "fsdp":
        if fsdp_axis <= 1:
            raise ValueError(
                "runtime.backend='fsdp' requires spmd.mesh fsdp axis > 1.\n"
                f"  fsdp axis size: {fsdp_axis}"
            )

        unsupported_axes = {
            "tensor": tensor_axis,
            "sequence": sequence_axis,
            "pipeline": pipeline_axis,
        }

        active_unsupported_axes = {
            name: size
            for name, size in unsupported_axes.items()
            if size > 1
        }

        if active_unsupported_axes:
            raise ValueError(
                "runtime.backend='fsdp' currently supports pure FSDP only.\n"
                f"  active unsupported axes: {active_unsupported_axes}\n"
                "Use a future runtime.backend='parallel3d' for tensor or "
                "sequence parallel layouts."
            )


def _validate_attention_mesh_compatibility(config: LaughLMConfig) -> None:
    """
    Validate attention implementation against the selected backend/mesh.

    Current known limitation:
    - GSPMD SplashAttention shard_map path requires an active "data" mesh axis.
    - A pure fsdp=8 mesh has axis_names=("fsdp",), because data=1 is removed.
    - Therefore runtime.backend='fsdp' + attention_impl='splash' requires
      spmd.mesh.axis_sizes()["data"] > 1 for now.

    This fails early at config-load time instead of during TPU model init.
    """

    backend = str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )

    attention_impl = str(
        getattr(
            config.architecture,
            "attention_impl",
            "standard",
        )
    )

    if backend != "fsdp":
        return

    if attention_impl != "splash":
        return

    axis_sizes = config.spmd.mesh.axis_sizes()

    data_axis = int(
        axis_sizes.get(
            "data",
            1,
        )
    )

    if data_axis <= 1:
        raise ValueError(
            "runtime.backend='fsdp' with attention_impl='splash' requires "
            "an active 'data' mesh axis for the current GSPMD Splash "
            "shard_map path.\n"
            f"  data axis size: {data_axis}\n"
            f"  fsdp axis size: {axis_sizes.get('fsdp', 1)}\n"
            "Use a hybrid mesh such as data=2/fsdp=4 or data=4/fsdp=2. "
            "For pure fsdp=8, set architecture.attention_impl='standard'."
        )


# ------------------------------------------------------------
# Validation Rules
# ------------------------------------------------------------

def _validate_active_llama_architecture(config: LaughLMConfig) -> None:
    """Reject schema variants the maintained LLaMA model does not implement.

    The shared schema intentionally contains options reserved for future model
    families.  The maintained training path always constructs the LLaMA model,
    so accepting a value that the implementation ignores would make the YAML
    describe a different model from the one that is trained.
    """

    arch = config.architecture
    required_values = {
        "positional": (arch.positional, "rope"),
        "normalization": (arch.normalization, "rms_norm"),
        "norm_placement": (arch.norm_placement, "pre"),
        "ffn_type": (arch.ffn_type, "swiglu"),
        "residual": (arch.residual, "standard"),
        "embeddings": (arch.embeddings, "standard"),
    }

    unsupported = [
        f"{name}={actual!r} (supported: {expected!r})"
        for name, (actual, expected) in required_values.items()
        if actual != expected
    ]

    if arch.attention_variant == "mla":
        unsupported.append(
            "attention_variant='mla' (not implemented by LLaMA attention)"
        )

    if unsupported:
        raise ValueError(
            "The maintained LLaMA training path does not implement the "
            "following architecture option(s):\n"
            + "\n".join(f"  - {item}" for item in unsupported)
            + "\nUse the documented LLaMA options, or implement and validate "
            "the variant before enabling it."
        )


def _validate_attention_variant(config: LaughLMConfig) -> None:
    """Ensure attention labels agree with the Q/KV head geometry."""

    variant = config.architecture.attention_variant
    num_heads = config.model.num_heads
    num_kv_heads = config.model.num_kv_heads

    if variant == "mha":
        if num_kv_heads not in {None, num_heads}:
            raise ValueError(
                "attention_variant='mha' requires model.num_kv_heads to be "
                f"unset or equal to num_heads ({num_heads}), got {num_kv_heads}."
            )
        return

    if variant == "mqa" and num_kv_heads != 1:
        raise ValueError(
            "attention_variant='mqa' requires model.num_kv_heads=1, got "
            f"{num_kv_heads}."
        )

    if (
        config.architecture.attention_impl == "splash"
        and variant in {"gqa", "mqa"}
        and num_kv_heads != num_heads
    ):
        raise ValueError(
            "attention_impl='splash' does not yet support grouped KV heads "
            "in this LLaMA implementation. Use attention_variant='mha' with "
            f"num_kv_heads={num_heads}, or use attention_impl='standard' or "
            "'xla' until M5 adds real GQA/MQA Splash support."
        )

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

    Equal Q/KV head counts remain temporarily valid for backward-compatible
    configurations. The production configuration uses attention_variant='mha'
    until real GQA is implemented in M5.
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
