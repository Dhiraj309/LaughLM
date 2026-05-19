"""
LaughLM/data/memmap_loader.py

Frontier-grade memmap token loader.

Features
--------
1. Multi-host deterministic shard partitioning
2. Host-local batch semantics
3. Async prefetching
4. Vectorized sampling
5. Zero-copy memmap reads
6. TPU-safe numpy output
7. Infinite iterator
8. Deterministic RNG streams

Designed for:
- TPU v5e
- multi-host JAX
- GSPMD training
"""

from __future__ import annotations

import queue
import threading

from typing import List

import numpy as np


# ============================================================
# Async prefetch iterator
# ============================================================

class _PrefetchIterator:

    def __init__(
        self,
        generator,
        prefetch_size: int = 8,
    ):

        self.queue = queue.Queue(
            maxsize=prefetch_size
        )

        self._stop = object()

        def worker():

            try:

                for item in generator:

                    self.queue.put(item)

            finally:

                self.queue.put(self._stop)

        self.thread = threading.Thread(
            target=worker,
            daemon=True,
        )

        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):

        item = self.queue.get()

        if item is self._stop:
            raise StopIteration

        return item


# ============================================================
# Dataset
# ============================================================

class MemmapDataset:
    """
    Frontier-grade token memmap dataset.

    Parameters
    ----------
    paths:
        List of token shard paths

    seq_len:
        Sequence length

    global_batch_size:
        Total batch across ALL hosts/devices

    process_index:
        Current host/process index

    process_count:
        Total hosts/processes

    seed:
        Base RNG seed
    """

    def __init__(
        self,
        paths,
        seq_len: int,
        global_batch_size: int,
        seed: int = 42,
        process_index: int = 0,
        process_count: int = 1,
        prefetch_size: int = 8,
    ):

        if isinstance(paths, str):
            paths = [paths]

        if len(paths) == 0:

            raise ValueError(
                "No dataset shards provided"
            )

        self.process_index = process_index
        self.process_count = process_count

        self.global_batch_size = (
            global_batch_size
        )

        # ====================================================
        # IMPORTANT
        #
        # Each host receives LOCAL batch.
        #
        # Global batch =
        #   local_batch * process_count
        # ====================================================

        if (
            global_batch_size
            % process_count
            != 0
        ):
            raise ValueError(
                "global_batch_size must be divisible "
                "by process_count"
            )

        self.local_batch_size = (
            global_batch_size
            // process_count
        )

        # ====================================================
        # Deterministic host shard assignment
        # ====================================================

        assigned_paths = [
            p
            for i, p in enumerate(paths)
            if i % process_count == process_index
        ]

        # fallback if too few shards

        if len(assigned_paths) == 0:

            assigned_paths = paths

        self.paths = assigned_paths

        print(
            f"[dataset] host "
            f"{process_index}/{process_count}"
        )

        print(
            f"[dataset] assigned shards: "
            f"{len(self.paths)}"
        )

        self.shards = [
            np.memmap(
                p,
                dtype=np.uint16,
                mode="r",
            )
            for p in self.paths
        ]

        self.shard_lengths = [
            len(s)
            for s in self.shards
        ]

        self.total_tokens = int(
            sum(self.shard_lengths)
        )

        self.seq_len = seq_len

        # ====================================================
        # Independent deterministic RNG stream
        # ====================================================

        self.rng = np.random.default_rng(
            seed + process_index
        )

        # ====================================================
        # Cached offsets
        # ====================================================

        self._seq_offsets = np.arange(
            self.seq_len,
            dtype=np.int64,
        )

        print(
            f"[dataset] total tokens: "
            f"{self.total_tokens:,}"
        )

        print(
            f"[dataset] global batch: "
            f"{self.global_batch_size:,}"
        )

        print(
            f"[dataset] local batch: "
            f"{self.local_batch_size:,}"
        )

        self.prefetch_size = (
            prefetch_size
        )

    # ========================================================
    # Batch sampling
    # ========================================================

    def sample_batch(self):
        """
        Sample LOCAL batch.
        """

        batch_size = (
            self.local_batch_size
        )

        # ----------------------------------------------------
        # Sample shards
        # ----------------------------------------------------

        shard_ids = self.rng.integers(
            0,
            len(self.shards),
            size=batch_size,
        )

        lengths = np.take(
            self.shard_lengths,
            shard_ids,
        )

        max_offsets = (
            lengths
            - self.seq_len
            - 1
        )

        max_offsets = np.maximum(
            max_offsets,
            1,
        )

        # ----------------------------------------------------
        # Random start positions
        # ----------------------------------------------------

        offsets = (
            self.rng.random(batch_size)
            * max_offsets
        ).astype(np.int64)

        # ----------------------------------------------------
        # Vectorized indexing
        # ----------------------------------------------------

        indices = (
            offsets[:, None]
            + self._seq_offsets[None, :]
        )

        # ----------------------------------------------------
        # Allocate output
        # ----------------------------------------------------

        x = np.empty(
            (
                batch_size,
                self.seq_len,
            ),
            dtype=np.int32,
        )

        # ----------------------------------------------------
        # Grouped memmap gather
        # ----------------------------------------------------

        unique_shards, inverse = np.unique(
            shard_ids,
            return_inverse=True,
        )

        for local_idx, shard_id in enumerate(
            unique_shards
        ):

            mask = (
                inverse == local_idx
            )

            if not np.any(mask):
                continue

            shard_indices = indices[mask]

            x[mask] = self.shards[
                shard_id
            ][shard_indices]

        return np.ascontiguousarray(
            x,
            dtype=np.int32,
        )

    # ========================================================
    # Infinite iterator
    # ========================================================

    def _iterator(self):

        while True:

            yield self.sample_batch()

    def __iter__(self):

        return _PrefetchIterator(
            self._iterator(),
            prefetch_size=(
                self.prefetch_size
            ),
        )
