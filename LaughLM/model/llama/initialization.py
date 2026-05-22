"""
LaughLM/model/llama/initialization.py

Frontier-grade parameter initialization +
logical partition metadata.
"""

from __future__ import annotations

from flax import linen as nn

import jax
import jax.numpy as jnp


# ============================================================
# Initializers
# ============================================================

def llama_kernel_init(
    initializer_range: float,
):
    """
    Canonical Llama Gaussian init.
    """

    return nn.initializers.normal(
        stddev=initializer_range,
    )


def llama_bias_init():
    """
    Zero bias init.
    """

    return nn.initializers.zeros_init()


# ============================================================
# Logical axis helpers
# ============================================================

def get_dense_logical_axes(
    name: str,
):
    """
    Logical axes for Dense kernels.

    Kernel layout:
        [input_dim, output_dim]
    """

    # --------------------------------------------------------
    # Attention projections
    # --------------------------------------------------------

    #
    # Fused QKV projection
    #
    # [embed, qkv]
    #

    if name == "qkv_proj":
        return ("embed", "qkv")

    #
    # Legacy split projections
    #

    if name == "q_proj":
        return ("embed", "heads")

    if name in {
        "k_proj",
        "v_proj",
    }:
        return ("embed", "kv_heads")

    if name == "o_proj":
        return ("heads", "embed")

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    if name in {
        "gate_proj",
        "up_proj",
    }:
        return ("embed", "mlp")

    if name == "down_proj":
        return ("mlp", "embed")

    # --------------------------------------------------------
    # LM head
    # --------------------------------------------------------

    if name == "lm_head":
        return ("embed", "vocab")

    # --------------------------------------------------------
    # Generic fallback
    #
    # IMPORTANT:
    # Logical axis names must be UNIQUE per tensor.
    # Never use ("embed", "embed").
    # --------------------------------------------------------

    return ("input", "output")


# ============================================================
# Activation constraints
# ============================================================

def constrain_hidden_states(
    hidden_states,
):
    """
    [batch, sequence, embed]
    """

    return nn.with_logical_constraint(
        hidden_states,
        (
            "batch",
            "sequence",
            "embed",
        ),
    )


def constrain_mlp_activations(
    hidden_states,
):
    """
    [batch, sequence, mlp]
    """

    return nn.with_logical_constraint(
        hidden_states,
        (
            "batch",
            "sequence",
            "mlp",
        ),
    )


def constrain_attention_q(
    hidden_states,
):
    """
    [batch, heads, sequence, head_dim]
    """

    return nn.with_logical_constraint(
        hidden_states,
        (
            "batch",
            "heads",
            "sequence",
            None,
        ),
    )


def constrain_attention_kv(
    hidden_states,
):
    """
    [batch, kv_heads, sequence, head_dim]
    """

    return nn.with_logical_constraint(
        hidden_states,
        (
            "batch",
            "kv_heads",
            "sequence",
            None,
        ),
    )


# ============================================================
# Dense factory
# ============================================================

def create_dense(
    *,
    features: int,
    config,
    use_bias: bool,
    name: str,
):
    """
    Frontier-grade Dense factory.
    """

    logical_axes = (
        get_dense_logical_axes(name)
    )

    kernel_init = (
        nn.with_logical_partitioning(
            llama_kernel_init(
                config.initializer_range
            ),
            logical_axes,
        )
    )

    if use_bias:

        bias_axes = (
            logical_axes[-1],
        )

        bias_init = (
            nn.with_logical_partitioning(
                llama_bias_init(),
                bias_axes,
            )
        )

    else:

        bias_init = llama_bias_init()

    return nn.Dense(
        features=features,

        use_bias=use_bias,

        kernel_init=kernel_init,

        bias_init=bias_init,

        param_dtype=config.param_dtype,

        dtype=config.compute_dtype,

        precision=jax.lax.Precision.DEFAULT,

        name=name,
    )


# ============================================================
# Embedding factory
# ============================================================

def create_embedding(
    *,
    num_embeddings: int,
    features: int,
    config,
    name: str,
):
    """
    Frontier-grade embedding layer.

    Embedding layout:
        [vocab, embed]
    """

    embedding_init = (
        nn.with_logical_partitioning(
            llama_kernel_init(
                config.initializer_range
            ),
            (
                "vocab",
                "embed",
            ),
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
