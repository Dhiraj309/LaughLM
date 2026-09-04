"""Optional host-side training-integrity diagnostics.

These diagnostics are disabled by default because materializing parameter and
optimizer trees on the host is intentionally more expensive than normal step
logging. When enabled, callers should run them only at a sparse interval.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict

import jax
import numpy as np


def _tree_digest(tree: Any) -> Dict[str, Any]:
    digest = hashlib.sha256()
    squared_norm = 0.0
    nonfinite_leaves = 0
    leaves = 0
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if leaf is None:
            digest.update(f"{index}:None;".encode("utf-8"))
            continue
        array = np.asarray(jax.device_get(leaf))
        leaves += 1
        digest.update(
            f"{index}:{array.dtype}:{array.shape}:".encode("utf-8")
        )
        digest.update(array.tobytes(order="C"))
        if np.issubdtype(array.dtype, np.number):
            values = array.astype(np.float64, copy=False)
            finite = np.isfinite(values)
            if not np.all(finite):
                nonfinite_leaves += 1
            finite_values = values[finite]
            squared_norm += float(np.sum(finite_values * finite_values))
    return {
        "checksum": digest.hexdigest(),
        "l2_norm": math.sqrt(squared_norm),
        "leaf_count": leaves,
        "nonfinite_leaf_count": nonfinite_leaves,
    }


def state_integrity(params: Any, opt_state: Any) -> Dict[str, Any]:
    """Return checksums/norms used to prove state changes and finite updates."""
    parameter = _tree_digest(params)
    optimizer = _tree_digest(opt_state)
    return {
        "parameter_checksum": parameter["checksum"],
        "parameter_l2_norm": parameter["l2_norm"],
        "parameter_leaf_count": parameter["leaf_count"],
        "parameter_nonfinite_leaf_count": parameter["nonfinite_leaf_count"],
        "optimizer_state_checksum": optimizer["checksum"],
        "optimizer_state_l2_norm": optimizer["l2_norm"],
        "optimizer_state_leaf_count": optimizer["leaf_count"],
        "optimizer_state_nonfinite_leaf_count": optimizer["nonfinite_leaf_count"],
    }
