"""
LaughLM/training/train_state.py

Backend-neutral TrainState.

Design
------
- PMAP uses host-side token accounting in trainer.py.
- FSDP/GSPMD uses device-state token accounting through tokens_processed.
- tokens_processed is also preserved for checkpoint compatibility.

Important
---------
Do not put PMAP, FSDP, TP, SP, or MoE logic in this file.

This state object should remain small and backend-neutral.
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

    # Preserved for checkpoint compatibility and mesh/FSDP token accounting.
    #
    # PMAP trainer.py may still keep host-side token accounting as the
    # authoritative runtime value. That behavior is intentionally unchanged.
    tokens_processed: Any = 0

    rng_key: Any = None
    extra_state: Any = None

    def next_rng(self):
        """
        Split and advance the state's RNG key.

        Returns
        -------
        new_state:
            TrainState with updated rng_key.

        subkey:
            Fresh PRNG subkey.
        """

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
        """
        Return a new TrainState after one optimizer update.

        Parameters
        ----------
        params:
            Updated model parameters.

        opt_state:
            Updated optimizer state.

        tokens_in_step:
            Number of tokens processed by this optimizer update.

            If None:
                tokens_processed is preserved unchanged.

            If provided:
                tokens_processed is incremented by tokens_in_step.

        extra_state:
            Optional replacement for backend-specific auxiliary state.

        Notes
        -----
        This method intentionally does not know whether the caller is PMAP,
        FSDP, Parallel3D, or MoE.

        PMAP may use host-side token accounting.
        FSDP may use this device-state token accounting.
        """

        if tokens_in_step is None:
            new_tokens_processed = self.tokens_processed
        else:
            new_tokens_processed = self.tokens_processed + tokens_in_step

        return self.replace(
            params=params,
            opt_state=opt_state,
            step=self.step + 1,
            tokens_processed=new_tokens_processed,
            extra_state=(
                self.extra_state
                if extra_state is None
                else extra_state
            ),
        )


def create_train_state(
    *,
    params,
    optimizer,
    rng_key=None,
    extra_state=None,
):
    """
    Create a TrainState from params and optimizer.

    This helper is backend-neutral. Backend-specific placement, replication,
    sharding, and checkpoint restore should happen outside this function.
    """

    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=0,
        tokens_processed=0,
        rng_key=rng_key,
        extra_state=extra_state,
    )
