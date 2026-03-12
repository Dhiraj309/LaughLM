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

        self.tokens = np.concatenate(arrays)

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)

        print(f"Loaded dataset with {len(self.tokens):,} tokens")

    # ------------------------------------------------------------
    # Sample batch
    # ------------------------------------------------------------

    def sample_batch(self):

        ix = self.rng.integers(
            0,
            len(self.tokens) - self.seq_len - 1,
            size=self.batch_size
        )

        x = np.array(
            [self.tokens[i:i+self.seq_len] for i in ix],
            dtype=np.int32
        )

        return x

    # ------------------------------------------------------------
    # Infinite iterator
    # ------------------------------------------------------------

    def __iter__(self):

        while True:
            yield self.sample_batch()
