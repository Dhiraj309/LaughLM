import numpy as np
from typing import List


class MemmapDataset:
    """
    High-throughput dataset loader for token shards.

    Features
    --------
    • memory-mapped shards
    • random sequence sampling
    • multi-shard support
    • infinite iterator
    • vectorized batch sampling (no Python loops)
    """

    def __init__(
        self,
        paths,
        seq_len: int,
        batch_size: int,
        seed: int = 42,
    ):

        if isinstance(paths, str):
            paths = [paths]

        arrays = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        # NOTE: keeping existing behavior for now to avoid changing
        # dataset semantics. Later optimization phases may replace this
        # with true shard indexing instead of concatenation.
        self.tokens = np.concatenate(arrays)

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)

        # Precompute sequence offsets to avoid repeated allocations
        self._seq_offsets = np.arange(self.seq_len)

        print(f"Loaded dataset with {len(self.tokens):,} tokens")

    # ------------------------------------------------------------
    # Sample batch
    # ------------------------------------------------------------

    def sample_batch(self):
        """
        Sample a batch of sequences using vectorized indexing.

        This removes the Python loop previously used for slicing
        and allows NumPy to perform the operation in optimized C code.
        """

        # Random starting positions
        ix = self.rng.integers(
            0,
            len(self.tokens) - self.seq_len - 1,
            size=self.batch_size
        )

        # Build index matrix
        # shape: [batch_size, seq_len]
        indices = ix[:, None] + self._seq_offsets[None, :]

        # Vectorized gather
        x = self.tokens[indices]

        # Convert dtype for JAX embedding lookup
        return x.astype(np.int32)

    # ------------------------------------------------------------
    # Infinite iterator
    # ------------------------------------------------------------

    def __iter__(self):

        while True:
            yield self.sample_batch()
