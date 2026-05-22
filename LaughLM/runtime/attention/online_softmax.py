"""
LaughLM/runtime/attention/online_softmax.py
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def online_softmax_update(
    prev_max: Array,
    prev_sum: Array,
    prev_out: Array,
    block_logits: Array,
    block_values: Array,
):
    """
    Streaming softmax update.

    Parameters
    ----------
    prev_max:
        [..., T]

    prev_sum:
        [..., T]

    prev_out:
        [..., T, D]

    block_logits:
        [..., T, BK]

    block_values:
        [..., BK, D]
    """

    #
    # Local block max
    #

    block_max = jnp.max(
        block_logits,
        axis=-1,
    )

    #
    # Updated running max
    #

    new_max = jnp.maximum(
        prev_max,
        block_max,
    )

    #
    # Rescale previous statistics
    #

    prev_scale = jnp.exp(
        prev_max - new_max
    )

    block_scale = jnp.exp(
        block_max - new_max
    )

    #
    # Local exponentials
    #

    block_probs = jnp.exp(
        block_logits
        -
        block_max[..., None]
    )

    #
    # Local denominator
    #

    block_sum = jnp.sum(
        block_probs,
        axis=-1,
    )

    #
    # New denominator
    #

    new_sum = (
        prev_scale * prev_sum
        +
        block_scale * block_sum
    )

    #
    # Local output
    #

    block_out = jnp.einsum(
        "...tk,...kd->...td",
        block_probs,
        block_values,
    )

    #
    # Numerator update
    #

    prev_term = (
        prev_out
        *
        prev_sum[..., None]
        *
        prev_scale[..., None]
    )

    block_term = (
        block_out
        *
        block_scale[..., None]
    )

    new_out = (
        prev_term
        +
        block_term
    ) / new_sum[..., None]

    return (
        new_max,
        new_sum,
        new_out,
    )
