from typing import Dict, Any

from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# Parameter Estimation
# ------------------------------------------------------------

def estimate_parameters(config: LaughLMConfig) -> Dict[str, int]:
    """
    Estimate parameter counts based on architecture config.

    Returns
    -------
    dict containing parameter breakdown
    """

    d_model = config.model.d_model
    num_layers = config.model.num_layers
    vocab_size = config.model.vocab_size

    ffn_dim = 4 * d_model

    # embeddings
    embedding_params = vocab_size * d_model

    # attention parameters per layer
    attn_params = (
        d_model * d_model * 3   # QKV
        + d_model * d_model     # output projection
    )

    # MLP parameters per layer
    mlp_params = (
        d_model * ffn_dim
        + ffn_dim * d_model
    )

    # layer norms (approx)
    norm_params = 2 * d_model

    per_layer = attn_params + mlp_params + norm_params

    transformer_params = per_layer * num_layers

    total_params = embedding_params + transformer_params

    return {
        "embedding_params": embedding_params,
        "per_layer_params": per_layer,
        "transformer_params": transformer_params,
        "total_params": total_params
    }


# ------------------------------------------------------------
# FLOPs Estimation
# ------------------------------------------------------------

def estimate_flops_per_token(config: LaughLMConfig) -> float:
    """
    Estimate FLOPs per token using common transformer approximation.

    FLOPs ≈ 6 × parameters
    """

    params = estimate_parameters(config)["total_params"]

    return 6 * params


# ------------------------------------------------------------
# Memory Estimation
# ------------------------------------------------------------

def estimate_memory_usage(config: LaughLMConfig) -> Dict[str, float]:
    """
    Estimate training memory footprint.
    """

    params = estimate_parameters(config)["total_params"]

    # bf16 parameters
    param_memory = params * 2

    # Adam optimizer state
    optimizer_memory = params * 8

    # gradients
    grad_memory = params * 2

    total_memory = param_memory + optimizer_memory + grad_memory

    return {
        "parameter_memory_bytes": param_memory,
        "optimizer_memory_bytes": optimizer_memory,
        "gradient_memory_bytes": grad_memory,
        "total_memory_bytes": total_memory
    }


# ------------------------------------------------------------
# Training Step Estimation
# ------------------------------------------------------------

def estimate_training_steps(config: LaughLMConfig) -> Dict[str, Any]:
    """
    Estimate tokens per step and total steps.
    """

    seq_len = config.runtime.seq_len
    batch = config.runtime.micro_batch_per_device
    devices = config.parallelism.data_parallel

    tokens_per_step = seq_len * batch * devices

    total_tokens = config.runtime.total_tokens

    steps = total_tokens // tokens_per_step

    return {
        "tokens_per_step": tokens_per_step,
        "total_steps": steps
    }


# ------------------------------------------------------------
# Pre-flight Report
# ------------------------------------------------------------

def generate_preflight_report(config: LaughLMConfig) -> None:
    """
    Print a pre-training model report.
    """

    params = estimate_parameters(config)
    memory = estimate_memory_usage(config)
    steps = estimate_training_steps(config)

    print("\nModel Report")
    print("────────────────────────")

    print(f"Total parameters:        {params['total_params']:,}")
    print(f"Embedding parameters:    {params['embedding_params']:,}")
    print(f"Parameters per layer:    {params['per_layer_params']:,}")

    print("\nTraining Report")
    print("────────────────────────")

    print(f"Tokens per step:         {steps['tokens_per_step']:,}")
    print(f"Total training steps:    {steps['total_steps']:,}")
    print(f"Target tokens:           {config.runtime.total_tokens:,}")

    print("\nMemory Report")
    print("────────────────────────")

    print(f"Parameter memory:        {memory['parameter_memory_bytes'] / 1e9:.2f} GB")
    print(f"Optimizer memory:        {memory['optimizer_memory_bytes'] / 1e9:.2f} GB")
    print(f"Gradient memory:         {memory['gradient_memory_bytes'] / 1e9:.2f} GB")
    print(f"Estimated total:         {memory['total_memory_bytes'] / 1e9:.2f} GB")

    print()