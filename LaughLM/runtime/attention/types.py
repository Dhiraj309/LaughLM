"""
LaughLM/runtime/attention/types.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jax.numpy as jnp


Array = jnp.ndarray


class AttentionMode(str, Enum):
    TRAIN = "train"
    PREFILL = "prefill"
    DECODE = "decode"


class AttentionMaskType(str, Enum):
    CAUSAL = "causal"
    SLIDING = "sliding"
    CHUNK = "chunk"
    FULL = "full"


@dataclass
class AttentionMaskSpec:
    mask_type: AttentionMaskType
    sliding_window: Optional[int] = None
    chunk_size: Optional[int] = None


DEFAULT_MASK_VALUE = -1e30
