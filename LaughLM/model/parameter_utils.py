"""
LaughLM/model/parameter_utils.py

Parameter, FLOPs, and memory estimation for LaughLM.

PMAP production accuracy notes:
──────────────────────────────────────────────
1. Parameter estimates must match the real LLaMA config_factory path.
2. SwiGLU intermediate size uses the canonical LLaMA formula:
      int(8 * d_model / 3), rounded up to multiple_of=256.
3. num_kv_heads is taken from config.model.num_kv_heads or num_heads,
   matching build_llama_config() behavior.
4. Gradient accumulation is included in tokens_per_step.
5. Weight tying is respected: tied lm_head adds no separate params.
"""

from __future__ import annotations

from typing import Dict, Any

import jax

from LaughLM.config.schema import LaughLMConfig


# ────────────────────────────────────────────────────────────────
# Shared LLaMA sizing helpers
# ────────────────────────────────────────────────────────────────

def compute_llama_intermediate_size(
    d_model: int,
    multiple_of: int = 256,
) -> int:
    """
    Canonical LLaMA SwiGLU intermediate size.

    Formula:
        intermediate_size = int(8 * d_model / 3)
        rounded up to `multiple_of`

    This must stay aligned with:
        LaughLM/model/llama/config_factory.py
    """

    if d_model <= 0:
        raise ValueError("d_model must be > 0")

    if multiple_of <= 0:
        raise ValueError("multiple_of must be > 0")

    intermediate_size = int((8 * d_model) / 3)

    intermediate_size = (
        (intermediate_size + multiple_of - 1)
        // multiple_of
    ) * multiple_of

    return intermediate_size


# ────────────────────────────────────────────────────────────────
# Parameter Estimation
# ────────────────────────────────────────────────────────────────

def estimate_parameters(config: LaughLMConfig) -> Dict[str, int]:
    """
    Estimate parameter counts for the canonical LLaMA model path.

    Returns:
        embedding_params
        final_norm_params
        per_layer_params
        attn_params_per_layer
        mlp_params_per_layer
        norm_params_per_layer
        transformer_params
        lm_head_params
        non_embedding_params
        total_params
        intermediate_size
        num_kv_heads
    """

    d_model = int(config.model.d_model)
    num_layers = int(config.model.num_layers)
    num_heads = int(config.model.num_heads)
    vocab_size = int(config.model.vocab_size)

    if d_model <= 0:
        raise ValueError("config.model.d_model must be > 0")

    if num_layers <= 0:
        raise ValueError("config.model.num_layers must be > 0")

    if num_heads <= 0:
        raise ValueError("config.model.num_heads must be > 0")

    if vocab_size <= 0:
        raise ValueError("config.model.vocab_size must be > 0")

    if d_model % num_heads != 0:
        raise ValueError(
            "config.model.d_model must be divisible by "
            "config.model.num_heads"
        )

    # Match LlamaConfig behavior:
    # None means MHA, i.e. num_kv_heads == num_heads.
    num_kv_heads = (
        int(config.model.num_kv_heads)
        if config.model.num_kv_heads is not None
        else num_heads
    )

    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be > 0")

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_kv_heads"
        )

    head_dim = d_model // num_heads
    kv_dim = num_kv_heads * head_dim

    # ── Embeddings ────────────────────────────────────────────
    embedding_params = vocab_size * d_model

    # ── Attention per layer ───────────────────────────────────
    #
    # q_proj: d_model -> num_heads * head_dim == d_model
    # k_proj: d_model -> num_kv_heads * head_dim == kv_dim
    # v_proj: d_model -> num_kv_heads * head_dim == kv_dim
    # o_proj: d_model -> d_model
    #
    q_params = d_model * d_model
    k_params = d_model * kv_dim
    v_params = d_model * kv_dim
    o_params = d_model * d_model

    attn_params = (
        q_params
        + k_params
        + v_params
        + o_params
    )

    # ── MLP per layer ─────────────────────────────────────────
    #
    # Canonical LLaMA SwiGLU:
    # gate_proj: d_model -> intermediate_size
    # up_proj:   d_model -> intermediate_size
    # down_proj: intermediate_size -> d_model
    #
    intermediate_size = compute_llama_intermediate_size(
        d_model,
        multiple_of=256,
    )

    mlp_params = (
        d_model * intermediate_size
        + d_model * intermediate_size
        + intermediate_size * d_model
    )

    # ── Norms per layer ───────────────────────────────────────
    norm_type = config.architecture.normalization

    if norm_type == "rms_norm":
        # input_layernorm + post_attention_layernorm
        norm_params = 2 * d_model
    else:
        # LayerNorm has scale + bias.
        norm_params = 2 * (2 * d_model)

    per_layer = (
        attn_params
        + mlp_params
        + norm_params
    )

    transformer_params = per_layer * num_layers

    # ── LM head ───────────────────────────────────────────────
    if config.architecture.weight_tying:
        lm_head_params = 0
    else:
        lm_head_params = d_model * vocab_size

    # ── Final norm ────────────────────────────────────────────
    if norm_type == "rms_norm":
        final_norm_params = d_model
    else:
        final_norm_params = 2 * d_model

    total_params = (
        embedding_params
        + transformer_params
        + lm_head_params
        + final_norm_params
    )

    non_embedding_params = (
        total_params
        - embedding_params
    )

    return {
        "embedding_params": int(embedding_params),
        "final_norm_params": int(final_norm_params),
        "per_layer_params": int(per_layer),
        "attn_params_per_layer": int(attn_params),
        "mlp_params_per_layer": int(mlp_params),
        "norm_params_per_layer": int(norm_params),
        "transformer_params": int(transformer_params),
        "lm_head_params": int(lm_head_params),
        "non_embedding_params": int(non_embedding_params),
        "total_params": int(total_params),
        "intermediate_size": int(intermediate_size),
        "num_kv_heads": int(num_kv_heads),
    }


# ────────────────────────────────────────────────────────────────
# FLOPs Estimation
# ────────────────────────────────────────────────────────────────

def estimate_flops_per_token(config: LaughLMConfig) -> float:
    """
    Estimate PaLM-style non-embedding FLOPs per token.

    Approximation:
        6 × non_embedding_params

    Attention quadratic FLOPs are handled separately in logger.py.
    """

    params = estimate_parameters(config)

    return float(
        6
        * params["non_embedding_params"]
    )


# ────────────────────────────────────────────────────────────────
# Memory Estimation
# ────────────────────────────────────────────────────────────────

def _dtype_nbytes(dtype_name: str) -> int:
    if dtype_name in ("bfloat16", "float16"):
        return 2

    if dtype_name == "float32":
        return 4

    raise ValueError(
        f"Unsupported dtype for memory estimate: {dtype_name!r}"
    )


def estimate_memory_usage(config: LaughLMConfig) -> Dict[str, float]:
    """
    Estimate rough training memory footprint.

    This is intentionally approximate. It estimates replicated PMAP
    per-device state memory for:
      - params
      - Adam moments
      - gradients

    Activation memory is not included.
    """

    total_params = estimate_parameters(config)["total_params"]

    param_dtype = getattr(
        config.parallelism,
        "param_dtype",
        "float32",
    )

    param_bytes = _dtype_nbytes(param_dtype)

    mu_dtype = getattr(
        config.optimizer,
        "mu_dtype",
        "float32",
    )

    mu_bytes = _dtype_nbytes(mu_dtype)

    # Adam has first moment + second moment.
    # Optax mu_dtype controls first moment; second moment is treated as fp32.
    nu_bytes = 4

    parameter_memory = total_params * param_bytes
    optimizer_memory = total_params * (mu_bytes + nu_bytes)

    # In train_step.py gradients are accumulated/reduced in fp32.
    grad_memory = total_params * 4

    total_memory = (
        parameter_memory
        + optimizer_memory
        + grad_memory
    )

    return {
        "parameter_memory_bytes": float(parameter_memory),
        "optimizer_memory_bytes": float(optimizer_memory),
        "gradient_memory_bytes": float(grad_memory),
        "total_memory_bytes": float(total_memory),
    }


# ────────────────────────────────────────────────────────────────
# Training Step Estimation
# ────────────────────────────────────────────────────────────────

def estimate_training_steps(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> Dict[str, Any]:
    """
    Estimate tokens per step and total optimizer steps.

    Matches trainer.py:
        tokens_per_step =
            seq_len
            * micro_batch_per_device
            * num_devices
            * gradient_accumulation
    """

    if num_devices is None:
        num_devices = jax.device_count()

    seq_len = int(config.runtime.seq_len)
    batch = int(config.runtime.micro_batch_per_device)
    grad_accum = int(config.runtime.gradient_accumulation)

    if seq_len <= 0:
        raise ValueError("runtime.seq_len must be > 0")

    if batch <= 0:
        raise ValueError("runtime.micro_batch_per_device must be > 0")

    if grad_accum <= 0:
        raise ValueError("runtime.gradient_accumulation must be > 0")

    if num_devices <= 0:
        raise ValueError("num_devices must be > 0")

    tokens_per_step = (
        seq_len
        * batch
        * num_devices
        * grad_accum
    )

    total_tokens = int(config.runtime.total_tokens)

    steps = total_tokens // tokens_per_step

    return {
        "tokens_per_step": int(tokens_per_step),
        "total_steps": int(steps),
    }


# ────────────────────────────────────────────────────────────────
# Pre-flight Report
# ────────────────────────────────────────────────────────────────

def generate_preflight_report(
    config: LaughLMConfig,
    num_devices: int | None = None,
) -> None:
    """
    Print a pre-training model report.
    """

    params = estimate_parameters(config)
    memory = estimate_memory_usage(config)
    steps = estimate_training_steps(
        config,
        num_devices=num_devices,
    )

    print("\nModel Report")
    print("────────────────────────────────────────")
    print(f"  Total parameters:      {params['total_params']:,}")
    print(f"  Non-embedding params:  {params['non_embedding_params']:,}")
    print(f"  Embedding parameters:  {params['embedding_params']:,}")
    print(f"  Final norm params:     {params['final_norm_params']:,}")
    print(f"  Intermediate size:     {params['intermediate_size']:,}")
    print(f"  KV heads:              {params['num_kv_heads']:,}")
    print(f"  Per-layer parameters:  {params['per_layer_params']:,}")
    print(f"    Attention:           {params['attn_params_per_layer']:,}")
    print(f"    MLP:                 {params['mlp_params_per_layer']:,}")
    print(f"    Norms:               {params['norm_params_per_layer']:,}")
    print(f"  LM head parameters:    {params['lm_head_params']:,}")

    print("\nTraining Report")
    print("────────────────────────────────────────")
    print(f"  Tokens per step:       {steps['tokens_per_step']:,}")
    print(f"  Total training steps:  {steps['total_steps']:,}")
    print(f"  Target tokens:         {config.runtime.total_tokens:,}")

    print("\nMemory Report")
    print("────────────────────────────────────────")
    print(f"  Parameter memory:      {memory['parameter_memory_bytes'] / 1e9:.2f} GB")
    print(f"  Optimizer memory:      {memory['optimizer_memory_bytes'] / 1e9:.2f} GB")
    print(f"  Gradient memory:       {memory['gradient_memory_bytes'] / 1e9:.2f} GB")
    print(f"  Estimated total:       {memory['total_memory_bytes'] / 1e9:.2f} GB")
    print()