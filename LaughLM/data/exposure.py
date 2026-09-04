"""Streaming token-exposure statistics for recorded training provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def summarize_token_paths(
    paths: Iterable[str | Path],
    *,
    dtype: str | np.dtype,
    vocab_size: int,
    chunk_tokens: int = 1_000_000,
) -> dict[str, Any]:
    """Return exact vocabulary exposure counts using bounded memmap reads."""
    vocabulary = int(vocab_size)
    chunk = int(chunk_tokens)
    if vocabulary <= 0 or chunk <= 0:
        raise ValueError("vocab_size and chunk_tokens must be positive")

    storage_dtype = np.dtype(dtype)
    counts = np.zeros(vocabulary, dtype=np.int64)
    total_tokens = 0
    files: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Token shard does not exist: {path}")
        byte_size = int(path.stat().st_size)
        if byte_size % storage_dtype.itemsize:
            raise ValueError(f"Token shard is not aligned to {storage_dtype}: {path}")
        tokens = np.memmap(path, dtype=storage_dtype, mode="r")
        files.append(str(path))
        for start in range(0, len(tokens), chunk):
            values = np.asarray(tokens[start : start + chunk], dtype=np.int64)
            if values.size == 0:
                continue
            if int(values.min()) < 0 or int(values.max()) >= vocabulary:
                raise ValueError(
                    f"Token ID outside vocabulary range in {path}: "
                    f"min={int(values.min())}, max={int(values.max())}, vocab_size={vocabulary}"
                )
            counts += np.bincount(values, minlength=vocabulary)
            total_tokens += int(values.size)

    used = counts[counts > 0]
    return {
        "files": files,
        "total_tokens": int(total_tokens),
        "unique_token_count": int(used.size),
        "singleton_token_count": int(np.count_nonzero(counts == 1)),
        "repeated_token_count": int(np.count_nonzero(counts > 1)),
        "excess_token_exposure": int(total_tokens - used.size),
        "max_token_exposure": int(used.max()) if used.size else 0,
        "frequency_checksum": hashlib.sha256(counts.tobytes()).hexdigest(),
    }
