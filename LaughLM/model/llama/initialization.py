"""
LaughLM/model/llama/initialization.py

Canonical parameter initialization + logical sharding
utilities for Llama.

Frontier-grade SPMD additions:
────────────────────────────────────────────────────
1. Logical axis annotations on parameters
2. Tensor-parallel aware projection partitioning
3. FSDP-compatible embedding sharding
4. Activation logical constraints
5. Scan-compatible parameter semantics
6. GSPMD-ready initialization path

Design goals
------------
- HF-compatible initialization semantics
- deterministic initialization
- explicit dtype handling
- TPU-safe parameter initialization
- future-ready for:
    - FSDP
    - tensor parallelism
    - sequence parallelism
    - scan-over-layers
    - quantization
    - LoRA / PEFT

Logical tensor conventions
──────────────────────────
Embedding:
    ("vocab", "embed")

Attention:
    Q proj:
        ("embed", "heads")

    K/V proj:
        ("embed", "kv_heads")

    O proj:
        ("heads", "embed")

MLP:
    gate/up:
        ("embed", "mlp")

    down:
        ("mlp", "embed")
"""

from flax import linen as nn

import jax.numpy as jnp


DEFAULT_PARAM_DTYPE = jnp.float32


# ─────────────────────────────────────────────────────────────
# Initializers
# ─────────────────────────────────────────────────────────────

def llama_kernel_init(
    initializer_range: float,
):
    """
    Canonical Llama Gaussian kernel initializer.

    Equivalent to HF:
        normal_(std=config.initializer_range)
    """

    return nn.initializers.normal(
        stddev=initializer_range,
    )


def llama_bias_init():
    """
    Canonical zero bias initializer.
    """

    return nn.initializers.zeros_init()


# ─────────────────────────────────────────────────────────────
# Logical axis helpers
# ─────────────────────────────────────────────────────────────

def get_dense_logical_axes(
    name: str,
):
    """
    Return logical partition axes for Dense kernels.

    Returns
    -------
    tuple[str | None, str | None]
        Logical axes for:
            (input_dim, output_dim)
    """

    # --------------------------------------------------------
    # Attention projections
    # --------------------------------------------------------

    if name == "q_proj":
        return ("embed", "heads")

    if name == "k_proj":
        return ("embed", "kv_heads")

    if name == "v_proj":
        return ("embed", "kv_heads")

    if name == "o_proj":
        return ("heads", "embed")

    # --------------------------------------------------------
    # MLP projections
    # --------------------------------------------------------

    if name == "gate_proj":
        return ("embed", "mlp")

    if name == "up_proj":
        return ("embed", "mlp")

    if name == "down_proj":
        return ("mlp", "embed")

    # --------------------------------------------------------
    # LM head fallback
    # --------------------------------------------------------

    if name == "lm_head":
        return ("embed", "vocab")

    # --------------------------------------------------------
    # Default fallback
    # --------------------------------------------------------

    return ("embed", "embed")


# ─────────────────────────────────────────────────────────────
# Activation constraints
# ─────────────────────────────────────────────────────────────

def constrain_hidden_states(
    hidden_states,
):
    """
    Standard hidden-state sharding constraint.

    Shape:
        [batch, sequence, embed]
    """

    return nn.with_logical_constraint(
        hidden_states,
        ("batch", "sequence", "embed"),
    )


def constrain_mlp_activations(
    hidden_states,
):
    """
    MLP intermediate activation constraint.

    Shape:
        [batch, sequence, mlp]
    """

    return nn.with_logical_constraint(
        hidden_states,
        ("batch", "sequence", "mlp"),
    )


def constrain_attention_q(
    hidden_states,
):
    """
    Attention Q tensor constraint.

    Shape:
        [batch, heads, sequence, head_dim]
    """

    return nn.with_logical_constraint(
        hidden_states,
        ("batch", "heads", "sequence", None),
    )


def constrain_attention_kv(
    hidden_states,
):
    """
    Attention KV tensor constraint.

    Shape:
        [batch, kv_heads, sequence, head_dim]
    """

    return nn.with_logical_constraint(
        hidden_states,
        ("batch", "kv_heads", "sequence", None),
    )


# ─────────────────────────────────────────────────────────────
# Dense factory
# ─────────────────────────────────────────────────────────────

def create_dense(
    *,
    features: int,
    config,
    use_bias: bool,
    name: str,
):
    """
    Frontier-grade Dense factory with logical partitioning.

    TPU frontier policy:
    ───────────────────
    params:
        fp32

    compute:
        bf16

    outputs:
        bf16

    Sharding:
    ─────────
    kernels use logical axis annotations for:
        - tensor parallelism
        - FSDP
        - GSPMD propagation
    """

    logical_axes = get_dense_logical_axes(
        name
    )

    kernel_init = nn.with_logical_partitioning(
        llama_kernel_init(
            config.initializer_range
        ),
        logical_axes,
    )

    if use_bias:

        bias_init = nn.with_logical_partitioning(
            llama_bias_init(),
            (logical_axes[-1],),
        )

    else:

        bias_init = llama_bias_init()

    return nn.Dense(
        features=features,
        use_bias=use_bias,

        kernel_init=kernel_init,

        bias_init=bias_init,

        #
        # Frontier dtype policy
        #

        param_dtype=config.param_dtype,

        dtype=config.compute_dtype,

        name=name,
    )


# ─────────────────────────────────────────────────────────────
# Embedding factory
# ─────────────────────────────────────────────────────────────

def create_embedding(
    *,
    num_embeddings: int,
    features: int,
    config,
    name: str,
):
    """
    Canonical embedding layer factory.

    Logical axes:
        ("vocab", "embed")

    Enables:
        - vocab sharding
        - FSDP embedding partitioning
    """

    embedding_init = (
        nn.with_logical_partitioning(
            llama_kernel_init(
                config.initializer_range
            ),
            ("vocab", "embed"),
        )
    )

    return nn.Embed(
        num_embeddings=num_embeddings,
        features=features,

        embedding_init=embedding_init,

        param_dtype=config.param_dtype,

        dtype=config.compute_dtype,

        name=name,
    )
