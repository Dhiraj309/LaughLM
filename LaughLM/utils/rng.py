import jax
import jax.random as jrandom


class RNGManager:
    """
    Centralized RNG manager for LaughLM.

    Handles JAX PRNG key creation and deterministic splitting.
    """

    def __init__(self, seed: int):
        self._key = jrandom.PRNGKey(seed)

    @property
    def key(self):
        """Return current key (read-only)."""
        return self._key

    def next_key(self):
        """
        Split and return a new PRNG key.
        """

        self._key, subkey = jrandom.split(self._key)
        return subkey

    def split(self, num: int):
        """
        Generate multiple keys at once.
        """

        keys = jrandom.split(self._key, num + 1)
        self._key = keys[0]
        return keys[1:]

    def fold_in(self, data: int):
        """
        Fold additional data into RNG state.

        Useful for step-dependent randomness.
        """

        self._key = jrandom.fold_in(self._key, data)
        return self._key


def create_rng(seed: int) -> RNGManager:
    """
    Create a new RNG manager.

    Example
    -------
    rng = create_rng(42)
    key = rng.next_key()
    """

    return RNGManager(seed)
