"""
LaughLM/model/parameter_utils.py

Parameter, FLOPs, and memory estimation for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. GQA-aware parameter estimation — accounts for num_kv_heads < num_heads
   in QKV projection size. Old code assumed MHA (3 × d_model²).

2. SwiGLU/GEGLU-aware FFN estimation — uses 8/3 ratio with gate projection
   instead of hardcoded 4 × d_model. Matches compute_ffn_dim() in mlp.py.

3. Gradient accumulation in step estimation — tokens_per_step now includes
   gradient_accumulation factor. Old code omitted it, causing mismatch
   with the actual training loop.

4. Weight tying awareness — if weight_tying=True, lm_head params are zero.
"""

from typing import Dict, Any

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.layers.mlp import compute_ffn_dim


# ────────────────────────────────────────────────────────────────
# Parameter Estimation
# ────────────────────────────────────────────────────────────────

def estimate_parameters(config: LaughLMConfig) -> Dict[str, int]:
    """
    Estimate parameter counts accounting for GQA and SwiGLU.

    Returns
    -------
    dict with: embedding_params, per_layer_params, attn_params_per_layer,
               mlp_params_per_layer, transformer_params, lm_head_params,
               total_params
    """
    d_model    = config.model.d_model
    num_layers = config.model.num_layers
    num_heads  = config.model.num_heads
    vocab_size = config.model.vocab_size

    # ── GQA-aware KV heads ────────────────────────────────────
    variant = config.architecture.attention_variant
    if variant == "mha":
        num_kv_heads = num_heads
    elif variant == "mqa":
        num_kv_heads = 1
    elif variant == "gqa":
        num_kv_heads = config.model.num_kv_heads or num_heads
    else:
        num_kv_heads = num_heads

    head_dim = d_model // num_heads
    kv_dim = num_kv_heads * head_dim

    # ── Embeddings ────────────────────────────────────────────
    embedding_params = vocab_size * d_model

    # ── Attention per layer (GQA-aware) ───────────────────────
    # Fused QKV: d_model → (d_model + 2 * kv_dim)
    # Output:    d_model → d_model
    qkv_params = d_model * (d_model + 2 * kv_dim)
    out_params = d_model * d_model
    attn_params = qkv_params + out_params

    # ── MLP per layer (SwiGLU/GEGLU-aware) ────────────────────
    ffn_type = config.architecture.ffn_type
    ffn_dim = compute_ffn_dim(d_model, ffn_type, multiple_of=64)

    if ffn_type in ("swiglu", "geglu"):
        # Gate+Up fused: d_model → 2*ffn_dim, Down: ffn_dim → d_model
        mlp_params = d_model * (2 * ffn_dim) + ffn_dim * d_model
    else:
        # Standard: d_model → ffn_dim, ffn_dim → d_model
        mlp_params = d_model * ffn_dim + ffn_dim * d_model

    # ── Norms per layer ───────────────────────────────────────
    norm_type = config.architecture.normalization
    if norm_type == "rms_norm":
        norm_params = 2 * d_model        # 2 RMSNorms × d_model (scale only)
    else:
        norm_params = 2 * (2 * d_model)  # 2 LayerNorms × (scale + bias)

    per_layer = attn_params + mlp_params + norm_params
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

    total_params = embedding_params + transformer_params + lm_head_params + final_norm_params

    return {
        "embedding_params": embedding_params,
        "per_layer_params": per_layer,
        "attn_params_per_layer": attn_params,
        "mlp_params_per_layer": mlp_params,
        "transformer_params": transformer_params,
        "lm_head_params": lm_head_params,
        "total_params": total_params,
    }


# ────────────────────────────────────────────────────────────────
# FLOPs Estimation
# ────────────────────────────────────────────────────────────────

def estimate_flops_per_token(config: LaughLMConfig) -> float:
    """
    Estimate FLOPs per token.

    Standard approximation: 6 × non-embedding parameters per token.
    Covers forward (2N) + backward (4N).
    """
    params = estimate_parameters(config)
    non_emb = params["total_params"] - params["embedding_params"]
    return 6 * non_emb


# ────────────────────────────────────────────────────────────────
# Memory Estimation
# ────────────────────────────────────────────────────────────────

def estimate_memory_usage(config: LaughLMConfig) -> Dict[str, float]:
    """
    Estimate training memory footprint.

    Assumes bf16 params + fp32 optimizer states (Adam: 2 × fp32 moments).
    """
    params = estimate_parameters(config)["total_params"]

    param_memory = params * 2         # bf16
    optimizer_memory = params * 8     # Adam: 2 moments × fp32 (4 bytes each)
    grad_memory = params * 2          # bf16 gradients

    total_memory = param_memory + optimizer_memory + grad_memory

    return {
        "parameter_memory_bytes": param_memory,
        "optimizer_memory_bytes": optimizer_memory,
        "gradient_memory_bytes": grad_memory,
        "total_memory_bytes": total_memory,
    }


# ────────────────────────────────────────────────────────────────
# Training Step Estimation
# ────────────────────────────────────────────────────────────────

def estimate_training_steps(config: LaughLMConfig) -> Dict[str, Any]:
    """
    Estimate tokens per step and total steps.

    INCLUDES gradient accumulation in tokens_per_step — matches
    the actual training loop in trainer.py.
    """
    seq_len    = config.runtime.seq_len
    batch      = config.runtime.micro_batch_per_device
    devices    = config.parallelism.data_parallel
    grad_accum = config.runtime.gradient_accumulation

    tokens_per_step = seq_len * batch * devices * grad_accum

    total_tokens = config.runtime.total_tokens
    steps = total_tokens // tokens_per_step

    return {
        "tokens_per_step": tokens_per_step,
        "total_steps": steps,
    }


# ────────────────────────────────────────────────────────────────
# Pre-flight Report
# ────────────────────────────────────────────────────────────────

def generate_preflight_report(config: LaughLMConfig) -> None:
    """Print a pre-training model report."""

    params = estimate_parameters(config)
    memory = estimate_memory_usage(config)
    steps  = estimate_training_steps(config)

    print("\nModel Report")
    print("────────────────────────────────────────")
    print(f"  Total parameters:      {params['total_params']:,}")
    print(f"  Embedding parameters:  {params['embedding_params']:,}")
    print(f"  Per-layer parameters:  {params['per_layer_params']:,}")
    print(f"    Attention:           {params['attn_params_per_layer']:,}")
    print(f"    MLP:                 {params['mlp_params_per_layer']:,}")
    print(f"  LM head parameters:   {params['lm_head_params']:,}")

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