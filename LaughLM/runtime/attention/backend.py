"""
LaughLM/runtime/attention/backend.py
"""

from __future__ import annotations

from enum import Enum

from .flash import flash_attention
from .reference import reference_attention


class AttentionBackend(str, Enum):
    REFERENCE = "reference"
    FLASH = "flash"


def apply_attention(
    query,
    key,
    value,
    mask_spec,
    *,
    backend: AttentionBackend = AttentionBackend.FLASH,
    block_q: int = 128,
    block_kv: int = 128,
):
    """
    Unified attention runtime dispatcher.
    """

    if backend == AttentionBackend.REFERENCE:
        return reference_attention(
            query,
            key,
            value,
            mask_spec,
        )

    if backend == AttentionBackend.FLASH:
        return flash_attention(
            query,
            key,
            value,
            mask_spec,
            block_q=block_q,
            block_kv=block_kv,
        )

    raise ValueError(
        f"Unknown backend: {backend}"
    )
