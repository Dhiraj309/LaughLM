"""
LaughLM/data/memmap_loader.py

High-throughput dataset loader for token shards.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Multi-host awareness — process_index and process_count parameters
   allow each JAX process (host) to read a disjoint subset of shards.
   This prevents data duplication in multi-node training.

2. Seed offset per host — each host gets a unique random seed derived
   from (base_seed + process_index) to avoid correlated sampling.

3. Existing features preserved:
   - Memory-mapped shards for zero-copy reads
   - Vectorized batch sampling (no Python loops)
   - Infinite iterator
   - Shard-grouped IO for sequential access

Reference: MaxText uses grain-based data pipeline with per-host sharding.
For memmap datasets, per-host shard assignment is sufficient.
"""

import numpy as np
from typing import List, Optional


class MemmapDataset:
    """
    High-throughput dataset loader for pre-tokenized binary shards.

    Parameters
    ----------
    paths : str or list of str — paths to .bin shard files
    seq_len : int — sequence length per sample
    batch_size : int — GLOBAL batch size (micro_batch * num_devices)
    seed : int — random seed for reproducibility
    process_index : int — this host's index (0 for single-host)
    process_count : int — total number of hosts (1 for single-host)
    """

    def __init__(
        self,
        paths,
        seq_len: int,
        batch_size: int,
        seed: int = 42,
        process_index: int = 0,
        process_count: int = 1,
    ):
        if isinstance(paths, str):
            paths = [paths]

        # ── Multi-host shard assignment ───────────────────────
        # Each host gets a disjoint subset of shards. If fewer
        # shards than hosts, all hosts share all shards but use
        # different random seeds to avoid duplicate sequences.
        if process_count > 1 and len(paths) >= process_count:
            # Round-robin shard assignment
            paths = [p for i, p in enumerate(paths) if i % process_count == process_index]
            print(f"[dataset] Host {process_index}/{process_count}: assigned {len(paths)} shards")
        elif process_count > 1:
            print(f"[dataset] Host {process_index}/{process_count}: sharing all {len(paths)} shards (fewer shards than hosts)")

        self.shards = [
            np.memmap(p, dtype=np.uint16, mode="r")
            for p in paths
        ]

        self.shard_lengths = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lengths)

        self.seq_len = seq_len
        self.batch_size = batch_size

        # Per-host seed to avoid correlated sampling across hosts
        self.rng = np.random.default_rng(seed + process_index)

        # Precompute offsets (avoid realloc every step)
        self._seq_offsets = np.arange(self.seq_len)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        print(
            f"[dataset] {self.total_tokens:,} tokens across {len(self.shards)} shards"
        )
        print(f"[dataset] batch_size (GLOBAL): {self.batch_size:,}")

    def sample_batch(self):
        """Sample a random batch of sequences from shards."""

        # Sample which shard each row comes from
        shard_ids = self.rng.integers(
            0,
            len(self.shards),
            size=self.batch_size,
        )

        lengths = np.take(self.shard_lengths, shard_ids)

        # Avoid negative offsets
        max_offsets = lengths - self.seq_len - 1
        max_offsets = np.maximum(max_offsets, 1)

        # Vectorised random start positions
        ix = (self.rng.random(self.batch_size) * max_offsets).astype(np.int64)

        # Build full sequence indices
        indices = ix[:, None] + self._seq_offsets[None, :]

        # Group by shard (minimize random IO)
        unique_shards, inverse = np.unique(shard_ids, return_inverse=True)

        x = np.empty((self.batch_size, self.seq_len), dtype=np.uint16)

        for shard_idx, shard_id in enumerate(unique_shards):
            mask = (inverse == shard_idx)

            if not np.any(mask):
                continue

            shard_indices = indices[mask]

            # Fast gather from memmap
            x[mask] = self.shards[shard_id][shard_indices]

        # Contiguous + int32 for JAX
        return np.ascontiguousarray(x, dtype=np.int32)

    def __iter__(self):
        """Infinite iterator."""
        while True:
            yield self.sample_batch()