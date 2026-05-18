"""
LaughLM/training/train_state.py

Canonical mesh-native TrainState.

Design goals
------------
- immutable functional state
- GSPMD-native semantics
- optimizer-agnostic
- compile-safe
- checkpoint-safe
- future EMA compatibility
- future fp32-master-weight compatibility
- no pmap assumptions
"""

from __future__ import annotations

from typing import Any

import jax

from flax import struct


@struct.dataclass
class TrainState:
    """
    Global sharded training state.

    GSPMD semantics
    ─────────────────────────────────────────────
    Arrays may be:
    - replicated
    - fully sharded
    - partially sharded

    XLA handles collectives automatically.

    No pmap/pmean semantics exist.
    """

    # --------------------------------------------------------
    # Model state
    # --------------------------------------------------------

    params: Any

    opt_state: Any

    # --------------------------------------------------------
    # Training progress
    # --------------------------------------------------------

    step: int = 0

    tokens_processed: int = 0

    # --------------------------------------------------------
    # RNG state
    # --------------------------------------------------------

    rng_key: jax.Array | None = None

    # --------------------------------------------------------
    # Optional future extensions
    # --------------------------------------------------------

    #
    # Future:
    # - EMA weights
    # - fp32 master params
    # - grad scaler
    # - metrics accumulators
    #

    extra_state: Any = None

    # ========================================================
    # RNG helpers
    # ========================================================

    def next_rng(
        self,
    ):
        """
        Split global RNG stream.

        Returns
        -------
        new_state:
            Updated state

        subkey:
            Per-step RNG key
        """

        if self.rng_key is None:

            raise ValueError(
                "TrainState.rng_key is None"
            )

        new_key, subkey = jax.random.split(
            self.rng_key
        )

        return (
            self.replace(
                rng_key=new_key
            ),
            subkey,
        )

    # ========================================================
    # Optimizer step update
    # ========================================================

    def apply_grad_step(
        self,
        *,
        params,
        opt_state,
        tokens_in_step: int,
        extra_state: Any | None = None,
    ):
        """
        Apply optimizer step update.

        Parameters
        ----------
        params:
            Updated parameters

        opt_state:
            Updated optimizer state

        tokens_in_step:
            GLOBAL tokens processed

        extra_state:
            Optional auxiliary state
        """

        return self.replace(
            params=params,
            opt_state=opt_state,
            step=self.step + 1,
            tokens_processed=(
                self.tokens_processed
                + jax.lax.convert_element_type(
                    tokens_in_step,
                    self.tokens_processed.dtype
                    if hasattr(
                        self.tokens_processed,
                        "dtype",
                    )
                    else type(
                        self.tokens_processed
                    ),
                )
            ),
            extra_state=(
                self.extra_state
                if extra_state is None
                else extra_state
            ),
        )


# ============================================================
# Factory
# ============================================================

def create_train_state(
    *,
    params,
    optimizer,
    rng_key=None,
    extra_state=None,
):
    """
    Create initialized TrainState.
    """

    opt_state = optimizer.init(
        params
    )

    return TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        tokens_processed=0,
        rng_key=rng_key,
        extra_state=extra_state,
    )
