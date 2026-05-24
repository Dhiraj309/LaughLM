"""
LaughLM/training/train_state.py

PMAP TrainState.

tokens_processed is kept only for checkpoint compatibility.
PMAP runtime token accounting is host-side in trainer.py.
"""

from __future__ import annotations

from typing import Any

import jax
from flax import struct


@struct.dataclass
class TrainState:
    params: Any
    opt_state: Any

    step: Any = 0

    # Keep this field so old checkpoints with tokens_processed restore.
    # Do not rely on this for long-run PMAP token accounting.
    tokens_processed: Any = 0

    rng_key: Any = None
    extra_state: Any = None

    def next_rng(self):
        if self.rng_key is None:
            raise ValueError("TrainState.rng_key is None")

        new_key, subkey = jax.random.split(self.rng_key)
        return self.replace(rng_key=new_key), subkey

    def apply_grad_step(
        self,
        *,
        params,
        opt_state,
        tokens_in_step: Any | None = None,
        extra_state: Any | None = None,
    ):
        return self.replace(
            params=params,
            opt_state=opt_state,
            step=self.step + 1,
            extra_state=self.extra_state if extra_state is None else extra_state,
        )


def create_train_state(
    *,
    params,
    optimizer,
    rng_key=None,
    extra_state=None,
):
    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=0,
        tokens_processed=0,
        rng_key=rng_key,
        extra_state=extra_state,
    )