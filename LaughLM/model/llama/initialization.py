"""
LaughLM/model/llama/initialization.py

Canonical parameter initialization utilities for Llama.

Design goals
------------
- HF-compatible initialization semantics
- deterministic initialization
- explicit dtype handling
- TPU-safe parameter initialization
- future-ready for sharding/quantization

Initialization semantics
------------------------
Dense / embedding kernels:
    Normal(stddev=config.initializer_range)

Biases:
    zeros

RMSNorm:
    ones

Parameter policy
----------------
Parameters remain fp32 by default.

Activations may later run in bf16/fp16.
"""

from collections.abc import Sequence

from flax import linen as nn

import jax.numpy as jnp


DEFAULT_PARAM_DTYPE = jnp.float32


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


def create_dense(
    *,
    features: int,
    config,
    use_bias: bool,
    name: str,
):
    """
    Canonical Dense layer factory.

    Ensures:
    - deterministic kernel init
    - explicit param dtype
    - HF-compatible semantics
    """

    return nn.Dense(
        features=features,
        use_bias=use_bias,
        kernel_init=llama_kernel_init(
            config.initializer_range
        ),
        bias_init=llama_bias_init(),
        dtype=jnp.float32,
        param_dtype=DEFAULT_PARAM_DTYPE,
        name=name,
    )


def create_embedding(
    *,
    num_embeddings: int,
    features: int,
    config,
    name: str,
):
    """
    Canonical embedding layer factory.
    """

    return nn.Embed(
        num_embeddings=num_embeddings,
        features=features,
        embedding_init=llama_kernel_init(
            config.initializer_range
        ),
        dtype=jnp.float32,
        param_dtype=DEFAULT_PARAM_DTYPE,
        name=name,
    )
