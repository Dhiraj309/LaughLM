from flax import struct


@struct.dataclass
class TrainState:
    """
    Full training state stored in checkpoints.

    Using a dataclass ensures Orbax restores the correct
    PyTree structure for Optax optimizer state.
    """

    params: any
    opt_state: any
    step: int
    tokens_processed: int
    rng_key: any
