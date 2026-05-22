"""
LaughLM/runtime/attention/utils.py
"""

from __future__ import annotations

import jax.numpy as jnp


def repeat_kv_heads(
    x,
    repeats: int,
):
    return jnp.repeat(
        x,
        repeats,
        axis=2,
    )
