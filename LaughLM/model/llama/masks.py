# LaughLM/model/llama/masks.py

"""
Causal attention masks for Llama.

Design goals
------------
- deterministic decode semantics
- explicit KV visibility
- static-cache compatibility
- HF-compatible causal behavior

Mask convention
---------------
Visible positions:
    0.0

Masked positions:
    dtype minimum
"""

import jax.numpy as jnp


def mask_neg_inf(dtype):

    return jnp.finfo(dtype).min


def build_causal_mask(
    query_length: int,
    key_length: int,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Standard causal mask.

    Shapes
    ------
    Output:
        [1, 1, Tq, Tk]
    """

    neg_inf = mask_neg_inf(
        dtype
    )

    q_idx = jnp.arange(
        query_length
    )[:, None]

    k_idx = jnp.arange(
        key_length
    )[None, :]

    mask = k_idx <= q_idx

    mask = jnp.where(
        mask,
        0.0,
        neg_inf,
    )

    return mask.astype(dtype)[
        None,
        None,
        :,
        :,
    ]


def build_decode_mask(
    query_length: int,
    key_length: int,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Decode-time causal mask.

    Supports chunked decoding.

    Shapes
    ------
    Output:
        [1, 1, Tq, Tk]
    """

    neg_inf = mask_neg_inf(
        dtype
    )

    #
    # Existing cache length
    #

    cache_length = (
        key_length - query_length
    )

    q_idx = (
        jnp.arange(query_length)[:, None]
        + cache_length
    )

    k_idx = jnp.arange(
        key_length
    )[None, :]

    mask = k_idx <= q_idx

    mask = jnp.where(
        mask,
        0.0,
        neg_inf,
    )

    return mask.astype(dtype)[
        None,
        None,
        :,
        :,
    ]
