"""
LaughLM/model/llama/masks.py

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
    large negative value
"""

import jax.numpy as jnp


NEG_INF = -1e30


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

    q_idx = jnp.arange(query_length)[:, None]

    k_idx = jnp.arange(key_length)[None, :]

    mask = k_idx <= q_idx

    mask = jnp.where(
        mask,
        0.0,
        NEG_INF,
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
    Decode-time mask.

    During decode:
    - queries are newest tokens
    - keys are all visible cache tokens

    Therefore everything is visible.

    Shapes
    ------
    Output:
        [1, 1, Tq, Tk]
    """

    return jnp.zeros(
        (
            1,
            1,
            query_length,
            key_length,
        ),
        dtype=dtype,
    )