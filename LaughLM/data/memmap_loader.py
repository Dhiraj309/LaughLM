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

    IMPORTANT
    ---------
    batch_size MUST be GLOBAL batch size:
        = micro_batch_per_device * num_devices
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

        self.shards = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        self.shard_lengths = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lengths)

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)

        # ✅ PRECOMPUTE OFFSETS (avoid realloc every step)
        self._seq_offsets = np.arange(self.seq_len)

        # ✅ SANITY CHECK (prevents silent TPU underfeeding bug)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        print(
            f"Loaded dataset with {self.total_tokens:,} tokens "
            f"across {len(self.shards)} shards"
        )
        print(f"[dataset] batch_size (GLOBAL): {self.batch_size:,}")

    # ------------------------------------------------------------
    # Sample batch
    # ------------------------------------------------------------

    def sample_batch(self):

        # ✅ SAMPLE WHICH SHARD EACH ROW COMES FROM
        shard_ids = self.rng.integers(
            0,
            len(self.shards),
            size=self.batch_size,
        )

        lengths = np.take(self.shard_lengths, shard_ids)

        # ✅ FIX: avoid negative offsets (edge-case safety)
        max_offsets = lengths - self.seq_len - 1
        max_offsets = np.maximum(max_offsets, 1)

        # ✅ VECTORISED RANDOM START POSITIONS
        ix = (self.rng.random(self.batch_size) * max_offsets).astype(np.int64)

        # ✅ BUILD FULL SEQUENCE INDICES
        indices = ix[:, None] + self._seq_offsets[None, :]

        # GROUP BY SHARD (minimise random IO)
        unique_shards, inverse = np.unique(shard_ids, return_inverse=True)

        x = np.empty((self.batch_size, self.seq_len), dtype=np.uint16)

        for shard_idx, shard_id in enumerate(unique_shards):
            mask = (inverse == shard_idx)

            if not np.any(mask):
                continue

            shard_indices = indices[mask]

            # ✅ FAST GATHER FROM MEMMAP
            x[mask] = self.shards[shard_id][shard_indices]

        # ✅ CRITICAL: contiguous + int32 for JAX
        return np.ascontiguousarray(x, dtype=np.int32)

    # ------------------------------------------------------------
    # Infinite iterator
    # ------------------------------------------------------------

    def __iter__(self):

        while True:
            yield self.sample_batch()