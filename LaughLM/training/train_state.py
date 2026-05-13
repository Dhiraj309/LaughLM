from flax import struct
from typing import Any
import jax


@struct.dataclass
class TrainState:
    """
    Full training state stored in checkpoints.

    Design notes
    ------------
    - params / opt_state are replicated across devices (pmap)
    - rng_key is per-device (must be replicated before pmap)
    - step is global optimizer step
    - tokens_processed is global (not per-device)
    """

    # Core state
    params: Any
    opt_state: Any

    step: int
    tokens_processed: int

    # RNG (per-device key when pmapped)
    rng_key: Any

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    def next_rng(self):
        """
        Split RNG safely.

        Returns:
            new_state, subkey
        """
        new_key, subkey = jax.random.split(self.rng_key)
        return self.replace(rng_key=new_key), subkey

    def apply_grad_step(self, params, opt_state, tokens_in_step: int):
        """
        Standardized state update after optimizer step.
        """
        return self.replace(
            params=params,
            opt_state=opt_state,
            step=self.step + 1,
            tokens_processed=self.tokens_processed + tokens_in_step,
        )