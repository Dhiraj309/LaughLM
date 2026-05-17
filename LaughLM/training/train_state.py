from flax import struct
from typing import Any

import jax


@struct.dataclass
class TrainState:
    """
    Global sharded training state.

    Frontier/GSPMD semantics
    ──────────────────────────────────────────────
    - params are globally sharded arrays
    - opt_state is globally sharded
    - step is global optimizer step
    - tokens_processed is GLOBAL token count
    - rng_key is a single global RNG stream

    IMPORTANT
    ──────────────────────────────────────────────
    This state is NOT replicated.

    Under GSPMD:
      - arrays may be partitioned across mesh axes
      - optimizer states may be partitioned differently
      - collectives are inserted automatically by XLA

    No pmap/pmean semantics exist here.
    """

    # ------------------------------------------------------------
    # Core training state
    # ------------------------------------------------------------

    params: Any

    opt_state: Any

    step: int

    tokens_processed: int

    # ------------------------------------------------------------
    # RNG state
    # ------------------------------------------------------------

    rng_key: Any

    # ------------------------------------------------------------
    # RNG utilities
    # ------------------------------------------------------------

    def next_rng(self):
        """
        Split global RNG safely.

        Returns
        -------
        new_state:
            Updated TrainState

        subkey:
            RNG key for current step
        """

        new_key, subkey = jax.random.split(
            self.rng_key
        )

        return (
            self.replace(
                rng_key=new_key
            ),
            subkey,
        )

    # ------------------------------------------------------------
    # Optimizer update utility
    # ------------------------------------------------------------

    def apply_grad_step(
        self,
        *,
        params,
        opt_state,
        tokens_in_step: int,
    ):
        """
        Standardized optimizer-step update.

        Parameters
        ----------
        params:
            Updated model parameters

        opt_state:
            Updated optimizer state

        tokens_in_step:
            GLOBAL tokens processed this step
        """

        return self.replace(
            params=params,
            opt_state=opt_state,
            step=self.step + 1,
            tokens_processed=(
                self.tokens_processed
                + tokens_in_step
            ),
        )
