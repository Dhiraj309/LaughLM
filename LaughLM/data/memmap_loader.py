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
from pathlib import Path

from typing import List

import numpy as np


UINT16_VOCAB_LIMIT = 65_536


def storage_dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Return the on-disk token dtype required by a vocabulary size.

    Token IDs are zero-based, so a vocabulary of exactly 65,536 entries still
    fits in uint16 (maximum ID 65,535). Larger vocabularies use uint64 as the
    project storage contract. Training batches are converted to int32 after
    loading, matching the current trainer input contract.
    """
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")

    return np.dtype(
        np.uint16
        if vocab_size <= UINT16_VOCAB_LIMIT
        else np.uint64
    )


def _validate_shard(
    path: str,
    *,
    dtype: np.dtype,
    seq_len: int,
    vocab_size: int,
    validation_sample_size: int = 4096,
) -> tuple[np.memmap, int, int]:
    """Open one raw token shard and validate its structural contract."""
    shard_path = Path(path)
    if not shard_path.is_file():
        raise FileNotFoundError(f"Token shard does not exist: {shard_path}")

    byte_size = shard_path.stat().st_size
    if byte_size == 0:
        raise ValueError(f"Token shard is empty: {shard_path}")
    if byte_size % dtype.itemsize != 0:
        raise ValueError(
            f"Token shard byte size is not divisible by {dtype} item size: "
            f"path={shard_path}, bytes={byte_size}, itemsize={dtype.itemsize}"
        )

    token_count = byte_size // dtype.itemsize
    if token_count < seq_len + 1:
        raise ValueError(
            f"Token shard is shorter than seq_len + 1: path={shard_path}, "
            f"tokens={token_count}, required={seq_len + 1}"
        )

    tokens = np.memmap(shard_path, dtype=dtype, mode="r")
    sample_size = min(int(validation_sample_size), token_count)
    sample_indices = np.unique(
        np.concatenate(
            (
                np.arange(sample_size, dtype=np.int64),
                np.arange(
                    token_count - sample_size,
                    token_count,
                    dtype=np.int64,
                ),
            )
        )
    )
    sampled_tokens = tokens[sample_indices]
    max_token_id = int(np.max(sampled_tokens))
    if max_token_id >= vocab_size:
        raise ValueError(
            f"Token ID is outside vocabulary range: path={shard_path}, "
            f"sampled_max_id={max_token_id}, vocab_size={vocab_size}"
        )

    return tokens, int(token_count), max_token_id


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
        self._error = object()

        def worker():

            try:

                for item in generator:

                    self.queue.put(item)

            except BaseException as exc:
                self.queue.put((self._error, exc))

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

        if isinstance(item, tuple) and item and item[0] is self._error:
            raise RuntimeError(
                "Prefetch worker failed while producing a token batch."
            ) from item[1]

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
        vocab_size: int = 65_536,
    ):

        if isinstance(paths, str):
            paths = [paths]

        if len(paths) == 0:

            raise ValueError(
                "No dataset shards provided"
            )

        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}.")
        if process_count <= 0:
            raise ValueError(
                f"process_count must be > 0, got {process_count}."
            )
        if process_index < 0 or process_index >= process_count:
            raise ValueError(
                f"process_index must be in [0, {process_count}), "
                f"got {process_index}."
            )
        if global_batch_size <= 0:
            raise ValueError(
                f"global_batch_size must be > 0, got {global_batch_size}."
            )

        self.vocab_size = int(vocab_size)
        self.storage_dtype = storage_dtype_for_vocab_size(
            self.vocab_size
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

        if len(assigned_paths) == 0:
            raise ValueError(
                "Host received no token shards under deterministic assignment: "
                f"process_index={process_index}, process_count={process_count}, "
                f"available_shards={len(paths)}. "
                "Provide at least one shard per process or reduce process_count."
            )

        self.paths = assigned_paths

        print(
            f"[dataset] host "
            f"{process_index}/{process_count}"
        )

        print(
            f"[dataset] assigned shards: "
            f"{len(self.paths)}"
        )

        print(
            f"[dataset] storage dtype: {self.storage_dtype}, "
            f"vocab size: {self.vocab_size}"
        )

        validated = [
            _validate_shard(
                p,
                dtype=self.storage_dtype,
                seq_len=seq_len,
                vocab_size=self.vocab_size,
            )
            for p in self.paths
        ]
        self.shards = [item[0] for item in validated]
        self.shard_lengths = [item[1] for item in validated]

        for path, (_, token_count, max_token_id) in zip(
            self.paths,
            validated,
        ):
            print(
                f"[dataset] shard={path} tokens={token_count:,} "
                f"sampled_max_id={max_token_id}"
            )

        self.total_tokens = int(
            sum(self.shard_lengths)
        )

        self.seq_len = int(seq_len)

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
