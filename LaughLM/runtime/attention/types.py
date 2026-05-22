"""
LaughLM/runtime/attention/types.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

DEFAULT_MASK_VALUE = -1e30


class AttentionBackend(str, Enum):

    REFERENCE = "reference"

    FLASH = "flash"

    ONLINE = "online"

    DECODE = "decode"

    RAGGED = "ragged"

    PAGED = "paged"


class AttentionMaskType(str, Enum):

    CAUSAL = "causal"

    FULL = "full"

    SLIDING_WINDOW = "sliding_window"

    CHUNK = "chunk"


@dataclass
class AttentionMaskSpec:

    mask_type: AttentionMaskType

    sliding_window: Optional[int] = None

    chunk_size: Optional[int] = None
