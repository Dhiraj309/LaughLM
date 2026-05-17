
"""
LaughLM/model/llama/sampling.py

Sampling utilities for autoregressive generation.

Design goals
------------
- deterministic semantics
- HF-compatible sampling order
- batch-safe top-p filtering
- numerically stable probability handling
- reusable generation primitives

Sampling pipeline
-----------------
logits
→ temperature scaling
→ top-k filtering
→ top-p filtering
→ categorical sampling

Tensor conventions
------------------
logits:
    [B, V]

returns:
    [B]
"""

from typing import Optional

import jax
import jax.numpy as jnp


NEG_INF = -1e10


def top_k_filter(
    logits: jnp.ndarray,
    top_k: int,
) -> jnp.ndarray:
    """
    Apply top-k filtering to logits.

    Parameters
    ----------
    logits:
        [B, V]

    top_k:
        Number of highest-probability tokens to keep.

    Returns
    -------
    filtered_logits:
        [B, V]
    """

    vocab_size = logits.shape[-1]

    if top_k <= 0 or top_k >= vocab_size:
        return logits

    topk_values, topk_indices = jax.lax.top_k(
        logits,
        top_k,
    )

    filtered = jnp.full_like(
        logits,
        NEG_INF,
    )

    batch_indices = jnp.arange(
        logits.shape[0]
    )[:, None]

    filtered = filtered.at[
        batch_indices,
        topk_indices,
    ].set(topk_values)

    return filtered


def top_p_filter(
    logits: jnp.ndarray,
    top_p: float,
) -> jnp.ndarray:
    """
    Apply nucleus (top-p) filtering.

    Parameters
    ----------
    logits:
        [B, V]

    top_p:
        Cumulative probability threshold.

    Returns
    -------
    filtered_logits:
        [B, V]

    Notes
    -----
    Keeps the minimal set of tokens whose cumulative
    probability mass exceeds top_p.
    """

    if top_p >= 1.0:
        return logits

    # ──────────────────────────────────────────
    # Sort descending
    # ──────────────────────────────────────────

    sorted_indices = jnp.argsort(
        logits,
        axis=-1,
    )[:, ::-1]

    sorted_logits = jnp.take_along_axis(
        logits,
        sorted_indices,
        axis=-1,
    )

    # ──────────────────────────────────────────
    # Compute cumulative probabilities
    # ──────────────────────────────────────────

    sorted_probs = jax.nn.softmax(
        sorted_logits.astype(jnp.float32),
        axis=-1,
    )

    cumulative_probs = jnp.cumsum(
        sorted_probs,
        axis=-1,
    )

    # ──────────────────────────────────────────
    # Mask tokens ABOVE threshold
    #
    # Keep first token exceeding threshold.
    # HF-compatible behavior.
    # ──────────────────────────────────────────

    sorted_mask = cumulative_probs > top_p

    sorted_mask = sorted_mask.at[:, 1:].set(
        sorted_mask[:, :-1]
    )

    sorted_mask = sorted_mask.at[:, 0].set(
        False
    )

    sorted_logits = jnp.where(
        sorted_mask,
        NEG_INF,
        sorted_logits,
    )

    # ──────────────────────────────────────────
    # Scatter back to original vocab order
    # ──────────────────────────────────────────

    filtered_logits = jnp.full_like(
        logits,
        NEG_INF,
    )

    batch_indices = jnp.arange(
        logits.shape[0]
    )[:, None]

    filtered_logits = filtered_logits.at[
        batch_indices,
        sorted_indices,
    ].set(sorted_logits)

    return filtered_logits


def sample_next_token(
    logits: jnp.ndarray,
    rng: jnp.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> jnp.ndarray:
    """
    Sample next token IDs from logits.

    Parameters
    ----------
    logits:
        [B, V]

    rng:
        PRNG key

    temperature:
        0.0 = greedy decoding

    top_k:
        Top-k filtering parameter

    top_p:
        Nucleus sampling threshold

    Returns
    -------
    token_ids:
        [B]
    """

    # ──────────────────────────────────────────
    # Greedy decoding
    # ──────────────────────────────────────────

    if temperature == 0.0:
        return jnp.argmax(
            logits,
            axis=-1,
        ).astype(jnp.int32)

    # ──────────────────────────────────────────
    # Temperature scaling
    # ──────────────────────────────────────────

    logits = logits / temperature

    # ──────────────────────────────────────────
    # Top-k filtering
    # ──────────────────────────────────────────

    logits = top_k_filter(
        logits,
        top_k,
    )

    # ──────────────────────────────────────────
    # Top-p filtering
    # ──────────────────────────────────────────

    logits = top_p_filter(
        logits,
        top_p,
    )

    # ──────────────────────────────────────────
    # Categorical sampling
    # ──────────────────────────────────────────

    token_ids = jax.random.categorical(
        rng,
        logits.astype(jnp.float32),
        axis=-1,
    )

    return token_ids.astype(jnp.int32)