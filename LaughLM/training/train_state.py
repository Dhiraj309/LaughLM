
from flax import struct
from typing import Any


@struct.dataclass
class TrainState:
    """
    Full training state stored in checkpoints.
    """

    params: Any
    opt_state: Any
    step: int
    tokens_processed: int
    rng_key: Any
