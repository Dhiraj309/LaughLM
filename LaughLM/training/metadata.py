"""
LaughLM/training/metadata.py

Run + checkpoint metadata utilities.

Design goals
────────────
- zero hot-path overhead
- topology-aware metadata
- TPU/GPU compatible
- JSON-only serialization
- restore-safe metadata capture
- no dependency on TrainState internals
- future FSDP compatibility
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

from pathlib import Path
from typing import Any

import jax
import numpy as np


# ============================================================
# Git helpers
# ============================================================


def _safe_git_commit() -> str | None:

    try:

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

    except Exception:
        return None


# ============================================================
# Device helpers
# ============================================================


def _device_summary() -> list[dict[str, Any]]:

    out = []

    for d in jax.devices():

        out.append(
            {
                "id": d.id,
                "process_index": d.process_index,
                "platform": d.platform,
                "device_kind": d.device_kind,
            }
        )

    return out


# ============================================================
# Mesh metadata
# ============================================================


def build_mesh_metadata(mesh) -> dict[str, Any]:

    return {
        "axis_names": list(mesh.axis_names),
        "shape": tuple(int(x) for x in mesh.devices.shape),
    }


# ============================================================
# Runtime metadata
# ============================================================


def build_run_metadata(
    *,
    config,
    mesh,
    total_params: int,
    embedding_params: int,
) -> dict[str, Any]:

    return {
        "timestamp": float(time.time()),
        "git_commit": _safe_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "devices": _device_summary(),
        "mesh": build_mesh_metadata(mesh),
        "hardware": {
            "accelerator": config.hardware.accelerator,
            "type": config.hardware.type,
        },
        "dtype_policy": {
            "param_dtype": config.spmd.dtype.param_dtype,
            "compute_dtype": config.spmd.dtype.compute_dtype,
            "output_dtype": config.spmd.dtype.output_dtype,
        },
        "model": {
            "d_model": config.model.d_model,
            "num_layers": config.model.num_layers,
            "num_heads": config.model.num_heads,
            "num_kv_heads": config.model.num_kv_heads,
            "vocab_size": config.model.vocab_size,
            "max_seq_len": config.model.max_seq_len,
        },
        "parameters": {
            "total": int(total_params),
            "embedding": int(embedding_params),
            "non_embedding": int(
                total_params - embedding_params
            ),
        },
        "config": config.model_dump(),
    }


# ============================================================
# Checkpoint metadata
# ============================================================


def build_checkpoint_metadata(
    *,
    config,
    mesh,
    step: int,
    tokens_processed: int,
) -> dict[str, Any]:

    return {
        "step": int(step),
        "tokens_processed": int(tokens_processed),
        "timestamp": float(time.time()),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "device_count": int(jax.device_count()),
        "mesh": build_mesh_metadata(mesh),
        "hardware": {
            "accelerator": config.hardware.accelerator,
            "type": config.hardware.type,
        },
        "dtype_policy": {
            "param_dtype": config.spmd.dtype.param_dtype,
            "compute_dtype": config.spmd.dtype.compute_dtype,
            "output_dtype": config.spmd.dtype.output_dtype,
        },
    }


# ============================================================
# JSON helpers
# ============================================================


def write_json(
    path: str | Path,
    data: dict[str, Any],
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,
        )
