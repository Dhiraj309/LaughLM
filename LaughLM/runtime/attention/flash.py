"""
LaughLM/runtime/attention/flash.py
"""

from __future__ import annotations

from .online_softmax import (
    online_attention,
)

from .types import (
    AttentionMaskSpec,
)

Array = object


def flash_attention(
    query,
    key,
    value,
    mask_spec: AttentionMaskSpec,
    *,
    block_q: int = 128,
    block_kv: int = 128,
):
    """
    Flash attention backend wrapper.

    Current implementation:
        tiled online attention.

    Future implementations:
        - Pallas TPU kernel
        - Triton GPU kernel
        - cuDNN FlashAttention
        - Splash attention
    """

    return online_attention(
        query,
        key,
        value,
        mask_spec,
        block_q=block_q,
        block_kv=block_kv,
    )
