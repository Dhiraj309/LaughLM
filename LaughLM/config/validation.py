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
    _validate_positional(config)
    _validate_norm_residual_compatibility(config)
    _validate_moe_requirements(config)


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
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )


def _validate_positional(config: LaughLMConfig) -> None:
    """
    Validate positional embedding compatibility.
    """

    positional = config.architecture.positional

    if positional == "rope_scaled":
        # scaled RoPE requires rope
        raise ValueError(
            "rope_scaled requires rope positional embedding implementation"
        )


def _validate_norm_residual_compatibility(config: LaughLMConfig) -> None:
    """
    DeepNorm requires matching residual configuration.
    """

    norm = config.architecture.normalization
    residual = config.architecture.residual

    if norm == "deep_norm" and residual != "deep_norm":
        raise ValueError(
            "deep_norm normalization requires deep_norm residual scaling"
        )


def _validate_moe_requirements(config: LaughLMConfig) -> None:
    """
    Placeholder validation for MoE architecture.
    """

    if config.architecture.ffn_type == "moe":
        raise ValueError(
            "MoE architecture selected but MoE parameters not yet implemented"
        )
