"""Fixed-batch wrapper used only by the overfit smoke gate."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class FixedBatchDataLoader:
    """Repeat one captured batch while preserving the loader interface."""

    def __init__(self, loader: Any):
        batch = next(iter(loader))
        self.batch = np.asarray(batch).copy()
        self.batch_checksum = hashlib.sha256(self.batch.tobytes()).hexdigest()

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch.copy()

    def get_state(self):
        return {
            "mode": "fixed_batch_v1",
            "batch_checksum": self.batch_checksum,
        }
