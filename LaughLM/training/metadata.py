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

Scheduler/resume design
───────────────────────
runtime.total_tokens:
    Current cumulative stage stop target.

scheduler.horizon_tokens:
    Fixed LR schedule horizon.

This separation allows staged pretraining:

    1B -> 2B -> 5B -> 20B

without reshaping/restarting the LR curve.
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


# ============================================================
# Git helpers
# ============================================================

def _safe_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
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
                "id": int(d.id),
                "process_index": int(d.process_index),
                "platform": str(d.platform),
                "device_kind": str(d.device_kind),
            }
        )

    return out


# ============================================================
# Mesh metadata
# ============================================================

def build_mesh_metadata(mesh) -> dict[str, Any]:
    return {
        "axis_names": list(mesh.axis_names),
        "shape": tuple(
            int(x)
            for x in mesh.devices.shape
        ),
    }


# ============================================================
# Runtime / scheduler helpers
# ============================================================

def _tokens_per_step(
    *,
    config,
    num_devices: int,
) -> int:
    tokens_per_step = (
        int(config.runtime.seq_len)
        * int(config.runtime.micro_batch_per_device)
        * int(num_devices)
        * int(config.runtime.gradient_accumulation)
    )

    if tokens_per_step <= 0:
        raise ValueError(
            "Computed tokens_per_step <= 0 while building metadata."
        )

    return int(tokens_per_step)


def _scheduler_horizon_tokens(config) -> int:
    if getattr(config.scheduler, "type", None) == "continuation_decay":
        end_tokens = getattr(
            config.scheduler,
            "continuation_end_tokens",
            None,
        )
        if end_tokens is not None:
            return int(end_tokens)

    horizon_tokens = getattr(
        config.scheduler,
        "horizon_tokens",
        None,
    )

    if horizon_tokens is None:
        return int(
            config.runtime.total_tokens
        )

    return int(
        horizon_tokens
    )


def _runtime_metadata(
    *,
    config,
    num_devices: int,
) -> dict[str, Any]:
    tokens_per_step = _tokens_per_step(
        config=config,
        num_devices=num_devices,
    )

    runtime_total_steps = (
        int(config.runtime.total_tokens)
        // tokens_per_step
    )

    return {
        "seq_len": int(config.runtime.seq_len),
        "micro_batch_per_device": int(
            config.runtime.micro_batch_per_device
        ),
        "gradient_accumulation": int(
            config.runtime.gradient_accumulation
        ),
        "total_tokens": int(
            config.runtime.total_tokens
        ),
        "tokens_per_step": int(
            tokens_per_step
        ),
        "total_steps": int(
            runtime_total_steps
        ),
    }


def _scheduler_metadata(
    *,
    config,
    num_devices: int,
) -> dict[str, Any]:
    tokens_per_step = _tokens_per_step(
        config=config,
        num_devices=num_devices,
    )

    horizon_tokens = _scheduler_horizon_tokens(
        config
    )

    scheduler_total_steps = (
        horizon_tokens
        // tokens_per_step
    )

    return {
        "type": str(config.scheduler.type),
        "horizon_tokens": int(
            horizon_tokens
        ),
        "total_steps": int(
            scheduler_total_steps
        ),
        "warmup_steps": (
            None
            if config.scheduler.warmup_steps is None
            else int(config.scheduler.warmup_steps)
        ),
        "warmup_fraction": (
            None
            if config.scheduler.warmup_fraction is None
            else float(config.scheduler.warmup_fraction)
        ),
        "stable_fraction": float(
            config.scheduler.stable_fraction
        ),
        "decay_steps": (
            None
            if config.scheduler.decay_steps is None
            else int(config.scheduler.decay_steps)
        ),
        "min_lr_ratio": float(
            config.scheduler.min_lr_ratio
        ),
        "continuation_start_tokens": (
            None
            if config.scheduler.continuation_start_tokens is None
            else int(config.scheduler.continuation_start_tokens)
        ),
        "continuation_end_tokens": (
            None
            if config.scheduler.continuation_end_tokens is None
            else int(config.scheduler.continuation_end_tokens)
        ),
        "continuation_start_lr": (
            None
            if config.scheduler.continuation_start_lr is None
            else float(config.scheduler.continuation_start_lr)
        ),
        "continuation_end_lr": (
            None
            if config.scheduler.continuation_end_lr is None
            else float(config.scheduler.continuation_end_lr)
        ),
        "continuation_decay_type": str(
            config.scheduler.continuation_decay_type
        ),
    }


def _optimizer_metadata(config) -> dict[str, Any]:
    return {
        "type": str(config.optimizer.type),
        "learning_rate": float(
            config.optimizer.learning_rate
        ),
        "beta1": float(
            config.optimizer.beta1
        ),
        "beta2": float(
            config.optimizer.beta2
        ),
        "eps": float(
            config.optimizer.eps
        ),
        "weight_decay": float(
            config.optimizer.weight_decay
        ),
        "gradient_clip": float(
            config.optimizer.gradient_clip
        ),
        "mu_dtype": str(
            getattr(
                config.optimizer,
                "mu_dtype",
                "float32",
            )
        ),
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
    num_devices = int(
        jax.device_count()
    )

    return {
        "timestamp": float(
            time.time()
        ),
        "git_commit": _safe_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "process_index": int(
            jax.process_index()
        ),
        "process_count": int(
            jax.process_count()
        ),
        "device_count": int(
            num_devices
        ),
        "local_device_count": int(
            jax.local_device_count()
        ),
        "devices": _device_summary(),
        "mesh": build_mesh_metadata(
            mesh
        ),
        "hardware": {
            "accelerator": str(
                config.hardware.accelerator
            ),
            "type": str(
                config.hardware.type
            ),
        },
        "dtype_policy": {
            "param_dtype": str(
                config.spmd.dtype.param_dtype
            ),
            "compute_dtype": str(
                config.spmd.dtype.compute_dtype
            ),
            "output_dtype": str(
                config.spmd.dtype.output_dtype
            ),
        },
        "model": {
            "d_model": int(
                config.model.d_model
            ),
            "num_layers": int(
                config.model.num_layers
            ),
            "num_heads": int(
                config.model.num_heads
            ),
            "num_kv_heads": (
                None
                if config.model.num_kv_heads is None
                else int(config.model.num_kv_heads)
            ),
            "vocab_size": int(
                config.model.vocab_size
            ),
            "max_seq_len": int(
                config.model.max_seq_len
            ),
        },
        "architecture": {
            "positional": str(
                config.architecture.positional
            ),
            "normalization": str(
                config.architecture.normalization
            ),
            "attention_impl": str(
                config.architecture.attention_impl
            ),
            "attention_variant": str(
                config.architecture.attention_variant
            ),
            "parallel_block": bool(
                config.architecture.parallel_block
            ),
            "fused_qkv": bool(
                getattr(
                    config.architecture,
                    "fused_qkv",
                    False,
                )
            ),
            "weight_tying": bool(
                config.architecture.weight_tying
            ),
        },
        "runtime": _runtime_metadata(
            config=config,
            num_devices=num_devices,
        ),
        "optimizer": _optimizer_metadata(
            config
        ),
        "scheduler": _scheduler_metadata(
            config=config,
            num_devices=num_devices,
        ),
        "parameters": {
            "total": int(
                total_params
            ),
            "embedding": int(
                embedding_params
            ),
            "non_embedding": int(
                total_params
                - embedding_params
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
    allow_legacy_v2: bool = False,
) -> dict[str, Any]:
    num_devices = int(
        jax.device_count()
    )
    
    if not allow_legacy_v2:
        raise RuntimeError(
            "build_checkpoint_metadata() emits legacy "
            "'laughlm_pmap_checkpoint_v2' metadata and must not be used "
            "for new PMAP/FSDP checkpoints.\n"
            "Use CheckpointManager.build_metadata_from_config() for "
            "layout-aware v3 checkpoint metadata.\n"
            "Pass allow_legacy_v2=True only for explicit legacy migration "
            "or compatibility tests."
        )

    runtime = _runtime_metadata(
        config=config,
        num_devices=num_devices,
    )

    return {
        "format": "laughlm_pmap_checkpoint_v2",
        "step": int(
            step
        ),
        "tokens_processed": int(
            tokens_processed
        ),
        "timestamp": float(
            time.time()
        ),
        "process_index": int(
            jax.process_index()
        ),
        "process_count": int(
            jax.process_count()
        ),
        "device_count": int(
            num_devices
        ),
        "tokens_per_step": int(
            runtime["tokens_per_step"]
        ),
        "mesh": build_mesh_metadata(
            mesh
        ),
        "hardware": {
            "accelerator": str(
                config.hardware.accelerator
            ),
            "type": str(
                config.hardware.type
            ),
        },
        "dtype_policy": {
            "param_dtype": str(
                config.spmd.dtype.param_dtype
            ),
            "compute_dtype": str(
                config.spmd.dtype.compute_dtype
            ),
            "output_dtype": str(
                config.spmd.dtype.output_dtype
            ),
        },
        "model": {
            "d_model": int(
                config.model.d_model
            ),
            "num_layers": int(
                config.model.num_layers
            ),
            "num_heads": int(
                config.model.num_heads
            ),
            "num_kv_heads": (
                None
                if config.model.num_kv_heads is None
                else int(config.model.num_kv_heads)
            ),
            "vocab_size": int(
                config.model.vocab_size
            ),
            "max_seq_len": int(
                config.model.max_seq_len
            ),
        },
        "architecture": {
            "positional": str(
                config.architecture.positional
            ),
            "normalization": str(
                config.architecture.normalization
            ),
            "attention_impl": str(
                config.architecture.attention_impl
            ),
            "attention_variant": str(
                config.architecture.attention_variant
            ),
            "parallel_block": bool(
                config.architecture.parallel_block
            ),
            "fused_qkv": bool(
                getattr(
                    config.architecture,
                    "fused_qkv",
                    False,
                )
            ),
            "weight_tying": bool(
                config.architecture.weight_tying
            ),
        },
        "runtime": runtime,
        "optimizer": _optimizer_metadata(
            config
        ),
        "scheduler": _scheduler_metadata(
            config=config,
            num_devices=num_devices,
        ),
    }


# ============================================================
# JSON helpers
# ============================================================

def write_json(
    path: str | Path,
    data: dict[str, Any],
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,
        )
