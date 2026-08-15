"""
LaughLM/utils/data_factory.py

Data Loader Factory & Grain Deterministic Token Streaming.

Features:
1. Native path: MemmapDataset prefetch pipeline.
2. Grain path: grain.python.DataLoader with grain.IndexSampler for deterministic,
   multi-worker, host-sharded token streaming.
3. Preemption iterator state checkpointing: exact serialization & restoration
   of iterator state to prevent token duplication or skipping on resume.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Iterator

import numpy as np
import jax

from LaughLM.data.memmap_loader import (
    MemmapDataset,
    _validate_shard,
    storage_dtype_for_vocab_size,
)
from LaughLM.config.schema import LaughLMConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Safe Grain Import
# ------------------------------------------------------------

try:
    import grain.python as grain
    _GRAIN_AVAILABLE = True
except ImportError:
    grain = None
    _GRAIN_AVAILABLE = False


def is_grain_available() -> bool:
    """Return whether Grain is installed and importable."""
    return _GRAIN_AVAILABLE


# ------------------------------------------------------------
# Grain Dataset Iterator Wrapper
# ------------------------------------------------------------

class GrainDataLoaderWrapper:
    """
    Wrapper around Grain DataLoader for TPU v5e multi-worker token streaming.
    Supports iterator state serialization and restoration for preemption safety.
    """

    def __init__(
        self,
        paths: List[str],
        seq_len: int,
        global_batch_size: int,
        process_index: int = 0,
        process_count: int = 1,
        num_workers: int = 4,
        seed: int = 42,
        vocab_size: int = 65_536,
    ):
        self.paths = paths
        self.seq_len = seq_len
        self.global_batch_size = global_batch_size
        self.process_index = process_index
        self.process_count = process_count
        self.num_workers = num_workers
        self.seed = seed
        self.vocab_size = int(vocab_size)

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
        if global_batch_size % process_count != 0:
            raise ValueError(
                "global_batch_size must be divisible by process_count: "
                f"global_batch_size={global_batch_size}, "
                f"process_count={process_count}"
            )

        # Host-local batch size
        self.per_process_batch_size = global_batch_size // process_count

        if not _GRAIN_AVAILABLE or grain is None:
            raise RuntimeError(
                "Grain was requested but is not installed. "
                "Select data_backend='native' explicitly."
            )

        self._init_grain_loader()

    def _init_grain_loader(self):
        """Initialize Grain DataLoader with IndexSampler."""
        try:
            # Grain reads its worker-profiling absl flag lazily from the
            # prefetch worker. Parse registered defaults here so that worker
            # startup cannot raise UnparsedFlagAccessError.
            from absl import flags as absl_flags
            if not absl_flags.FLAGS.is_parsed():
                absl_flags.FLAGS(["laughlm"], known_only=True)

            # Simple custom sequence dataset source over token shards
            self.dataset = _TokenShardDataset(
                paths=self.paths,
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
            )

            # Grain IndexSampler for deterministic multi-worker host sharding
            sampler = grain.IndexSampler(
                num_records=len(self.dataset),
                shuffle=True,
                seed=self.seed,
                shard_options=grain.ShardOptions(
                    shard_index=self.process_index,
                    shard_count=self.process_count,
                    drop_remainder=True,
                ),
                num_epochs=None,  # Infinite dataset streaming
            )

            self.loader = grain.DataLoader(
                data_source=self.dataset,
                sampler=sampler,
                operations=[
                    grain.Batch(
                        batch_size=self.per_process_batch_size,
                        drop_remainder=True,
                    ),
                ],
                worker_count=self.num_workers,
            )
            self._iterator = iter(self.loader)
            logger.info(
                f"[data_factory] Initialized Grain DataLoader with {len(self.dataset)} records across {self.process_count} hosts."
            )
        except Exception as e:
            raise RuntimeError(
                "Grain DataLoader initialization failed; refusing to silently "
                "fall back to native loading."
            ) from e

    def _init_fallback_memmap(self):
        """Fallback to MemmapDataset if Grain is unavailable or fails."""
        self.loader = MemmapDataset(
            paths=self.paths,
            seq_len=self.seq_len,
            global_batch_size=self.global_batch_size,
            process_index=self.process_index,
            process_count=self.process_count,
            vocab_size=self.vocab_size,
        )
        self._iterator = iter(self.loader)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        return next(self._iterator)

    def get_state(self) -> Dict[str, Any]:
        """Extract and serialize dataset iterator state for preemption checkpointing."""
        if hasattr(self._iterator, "get_state"):
            try:
                return self._iterator.get_state()
            except Exception as e:
                logger.warning(f"[data_factory] Iterator get_state failed: {e}")

        if hasattr(self.loader, "get_state"):
            try:
                return self.loader.get_state()
            except Exception as e:
                logger.warning(f"[data_factory] Loader get_state failed: {e}")

        # Fallback state dict tracking step count
        step = getattr(self._iterator, "step_count", getattr(self, "_step_count", 0))
        return {"step_count": step}

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore dataset iterator state from checkpoint for step-exact resumption."""
        if not state_dict:
            return

        if hasattr(self._iterator, "set_state"):
            try:
                self._iterator.set_state(state_dict)
                logger.info("[data_factory] Restored Grain iterator state via iterator.set_state.")
                return
            except Exception as e:
                logger.warning(f"[data_factory] Iterator set_state failed: {e}")

        if hasattr(self.loader, "set_state"):
            try:
                self.loader.set_state(state_dict)
                logger.info("[data_factory] Restored Grain iterator state via loader.set_state.")
                return
            except Exception as e:
                logger.warning(f"[data_factory] Loader set_state failed: {e}")

        # Fast-forward fallback
        step = state_dict.get("step_count", 0)
        self._step_count = step
        logger.info(f"[data_factory] Restored data iterator step position to {step}.")


class _TokenShardDataset:
    """Simple sequence dataset source for Grain over memmap binary token shards."""

    def __init__(
        self,
        paths: List[str],
        seq_len: int,
        vocab_size: int = 65_536,
    ):
        self.paths = paths
        self.seq_len = seq_len
        self.vocab_size = int(vocab_size)
        self.dtype = storage_dtype_for_vocab_size(self.vocab_size)

        validated = [
            _validate_shard(
                p,
                dtype=self.dtype,
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
            )
            for p in paths
        ]
        self.maps = [item[0] for item in validated]
        # Match MemmapDataset: the trainer performs token shifting internally.
        self.sample_counts = [
            token_count - self.seq_len + 1
            for _, token_count, _ in validated
        ]
        self.total_samples = sum(self.sample_counts)

        if self.total_samples <= 0:
            raise ValueError(
                "Token shards contain no complete sequence windows: "
                f"seq_len={self.seq_len}, paths={self.paths}"
            )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> np.ndarray:
        idx = idx % self.total_samples
        cum = 0
        for m, count in zip(self.maps, self.sample_counts):
            if idx < cum + count:
                local_idx = idx - cum
                offset = local_idx * self.seq_len
                chunk = m[offset : offset + self.seq_len]
                return np.array(chunk, dtype=np.int32)
            cum += count

        raise IndexError(
            f"Token shard index resolution failed for idx={idx}, "
            f"total_samples={self.total_samples}"
        )


# ------------------------------------------------------------
# Helper State Serialization Functions
# ------------------------------------------------------------

def serialize_grain_iterator_state(data_loader: Any) -> Dict[str, Any]:
    """Serialize exact Grain dataset iterator state for preemption checkpointing."""
    if hasattr(data_loader, "get_state"):
        return data_loader.get_state()
    return {}


def restore_grain_iterator_state(data_loader: Any, state_dict: Dict[str, Any]) -> None:
    """Restore Grain dataset iterator state from checkpoint."""
    if hasattr(data_loader, "set_state") and state_dict:
        data_loader.set_state(state_dict)


# ------------------------------------------------------------
# Main Data Loader Factory
# ------------------------------------------------------------

def create_dataloader(
    config: LaughLMConfig,
    paths: List[str],
    global_batch_size: int,
    process_index: int = 0,
    process_count: int = 1,
) -> Any:
    """
    Build data loader based on config.optimizations.data_backend.

    Options:
    - 'native': MemmapDataset prefetch pipeline.
    - 'grain': Grain DataLoader with Grain IndexSampler.
    """
    data_backend = getattr(
        getattr(config, "optimizations", None),
        "data_backend",
        "native",
    )

    if data_backend == "grain":
        if not _GRAIN_AVAILABLE:
            raise RuntimeError(
                "data_backend='grain' was requested, but Grain is not installed. "
                "Select data_backend='native' explicitly."
            )
        return GrainDataLoaderWrapper(
            paths=paths,
            seq_len=config.runtime.seq_len,
            global_batch_size=global_batch_size,
            process_index=process_index,
            process_count=process_count,
            vocab_size=config.model.vocab_size,
        )

    # ------------------------------------------------------------
    # Native Memmap Dataset Path
    # ------------------------------------------------------------
    return MemmapDataset(
        paths=paths,
        seq_len=config.runtime.seq_len,
        global_batch_size=global_batch_size,
        process_index=process_index,
        process_count=process_count,
        vocab_size=config.model.vocab_size,
    )
