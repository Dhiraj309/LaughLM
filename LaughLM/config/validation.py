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

    _validate_attention_heads(config)
    _validate_gqa_kv_heads(config)
    _validate_positional(config)
    _validate_norm_residual_compatibility(config)
    _validate_moe_requirements(config)
    _validate_wsd_scheduler(config)


# ------------------------------------------------------------
# Validation Rules
# ------------------------------------------------------------


def _validate_attention_heads(config: LaughLMConfig) -> None:
    """
    Ensure head dimension divides model dimension.
    """

    d_model   = config.model.d_model
    num_heads = config.model.num_heads

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"head_dim = {d_model} / {num_heads} = {d_model / num_heads:.2f} (not integer)."
        )


def _validate_gqa_kv_heads(config: LaughLMConfig) -> None:
    """
    When GQA is selected, num_kv_heads must:
      - Be specified (not None)
      - Divide num_heads evenly
      - Be <= num_heads
    """

    if config.architecture.attention_variant != "gqa":
        return

    num_heads    = config.model.num_heads
    num_kv_heads = config.model.num_kv_heads

    if num_kv_heads is None:
        raise ValueError(
            "attention_variant='gqa' requires model.num_kv_heads to be set. "
            f"For a {num_heads}-head model, typical values: "
            f"{num_heads // 4} (4:1) or {num_heads // 8} (8:1)."
        )

    if num_kv_heads > num_heads:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must be <= num_heads ({num_heads})."
        )

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) "
            f"so that each KV group covers the same number of Q heads."
        )


def _validate_positional(config: LaughLMConfig) -> None:
    """
    Validate positional embedding compatibility.
    ALiBi is not yet implemented. All others are supported.
    """

    positional = config.architecture.positional

    if positional == "alibi":
        raise ValueError(
            "positional='alibi' is not yet implemented. "
            "Use 'rope' (recommended) or 'learned'."
        )

    # rope_scaled is valid — no rejection needed.
    # Its sin/cos tables are computed identically to rope;
    # the scaling factor is applied at inference time via context extension.


def _validate_norm_residual_compatibility(config: LaughLMConfig) -> None:
    """
    DeepNorm requires matching residual configuration.
    """

    norm     = config.architecture.normalization
    residual = config.architecture.residual

    if norm == "deep_norm" and residual != "deep_norm":
        raise ValueError(
            "normalization='deep_norm' requires residual='deep_norm'. "
            "DeepNorm uses coordinated α/β scaling between norm and residual."
        )


def _validate_moe_requirements(config: LaughLMConfig) -> None:
    """
    Placeholder validation for MoE architecture.
    """

    if config.architecture.ffn_type == "moe":
        raise ValueError(
            "ffn_type='moe' selected but MoE is not yet implemented. "
            "Use 'swiglu' (recommended) or 'geglu'."
        )


def _validate_wsd_scheduler(config: LaughLMConfig) -> None:
    """
    WSD-specific cross-field validation.
    """

    if config.scheduler.type != "wsd":
        return

    stable_fraction = config.scheduler.stable_fraction

    if not (0.0 < stable_fraction < 1.0):
        raise ValueError(
            f"scheduler.stable_fraction must be in (0, 1), got {stable_fraction}."
        )

    warmup = config.scheduler.warmup_steps

    # Compute total steps for sanity check
    seq_len  = config.runtime.seq_len
    batch    = config.runtime.micro_batch_per_device
    devices  = config.parallelism.data_parallel
    grad_acc = config.runtime.gradient_accumulation

    tokens_per_step = seq_len * batch * devices * grad_acc
    total_steps     = config.runtime.total_tokens // tokens_per_step

    stable_steps = int(total_steps * stable_fraction)

    if warmup + stable_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup}) + stable_steps ({stable_steps}) "
            f">= total_steps ({total_steps}). "
            f"No steps remain for decay. Reduce stable_fraction or warmup_steps."
        )