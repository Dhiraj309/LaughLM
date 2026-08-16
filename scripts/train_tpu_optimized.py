from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _sanitize_single_vm_tpu_process_addresses() -> None:
    """Remove the invalid literal `local` TPU topology override before JAX import."""
    for env_name in ("TPU_PROCESS_ADDRESSES", "JAX_TPU_PROCESS_ADDRESSES"):
        value = os.environ.get(env_name)
        if value is not None and value.strip().lower() == "local":
            os.environ.pop(env_name, None)
            print(
                f"[train_tpu_optimized] Ignoring {env_name}=local for a single TPU VM; "
                "using JAX runtime device discovery instead.",
                flush=True,
            )


_sanitize_single_vm_tpu_process_addresses()

import jax

from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
# --- NEW IMPORTS ---
from LaughLM.utils.data_factory import create_dataloader
# -------------------


DEFAULT_CONFIG = "configs/v5e_pmap_optimized.yaml"


def _configure_persistent_compilation_cache(config) -> None:
    """Configure JAX persistent caching before the trainer compiles a step."""
    cache_dir = config.optimizations.compilation_cache_dir
    if not cache_dir:
        return

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    # Keep short TPU benchmarks cacheable and reuse compilation/autotuning work.
    # This runs before Trainer creates the PMAP-compiled train step.
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    print(
        "[train_tpu_optimized] persistent JAX compilation cache enabled:\n"
        f"  directory={cache_path}\n"
        "  min_compile_time_secs=0\n"
        "  min_entry_size_bytes=-1",
        flush=True,
    )


def _compilation_cache_snapshot(config) -> dict:
    """Capture cache state before this run changes it."""
    cache_dir = config.optimizations.compilation_cache_dir
    if not cache_dir:
        return {
            "configured": False,
            "directory": None,
            "exists_before_run": False,
            "file_count_before_run": 0,
        }

    cache_path = Path(cache_dir).expanduser().resolve()
    file_count = 0
    if cache_path.exists():
        try:
            file_count = sum(
                1
                for path in cache_path.rglob("*")
                if path.is_file()
            )
        except OSError:
            file_count = -1

    return {
        "configured": True,
        "directory": str(cache_path),
        "exists_before_run": cache_path.exists(),
        "file_count_before_run": file_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LaughLM with Optimized JAX Stack."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to optimized config YAML.",
    )

    parser.add_argument(
        "--override-config",
        type=str,
        default=None,
        help=(
            "Optional YAML overlay merged into --config. Use this for "
            "isolated architecture experiments without copying the baseline."
        ),
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override runtime.total_tokens for short benchmark runs.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete checkpoint_dir before training.",
    )

    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Override data.hf_repo_id for this run.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional Hugging Face dataset revision override.",
    )
    parser.add_argument(
        "--shard-directory",
        type=str,
        default=None,
        help="Override data.shard_directory, for example fineweb-edu.",
    )
    parser.add_argument(
        "--shard-filename-prefix",
        type=str,
        default=None,
        help="Override data.shard_filename_prefix, for example fineweb-edu_shard.",
    )
    parser.add_argument(
        "--train-shard-start",
        type=int,
        default=None,
        help="First training shard ID to download.",
    )
    parser.add_argument(
        "--train-shard-count",
        type=int,
        default=None,
        help="Number of consecutive training shard files to download.",
    )
    parser.add_argument(
        "--validation-shard-start",
        type=int,
        default=None,
        help="First held-out validation shard ID to download.",
    )
    parser.add_argument(
        "--validation-shard-count",
        type=int,
        default=None,
        help="Number of consecutive held-out validation shard files to download.",
    )

    return parser.parse_args()


def _tokens_per_step(config, *, num_devices: int) -> int:
    return int(
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * num_devices
        * config.runtime.gradient_accumulation
    )


def _apply_max_steps_override(
    config,
    *,
    max_steps: int | None,
    num_devices: int,
):
    if max_steps is None:
        return config

    if max_steps <= 0:
        raise ValueError(
            "--max_steps must be > 0"
        )

    tokens_per_step = _tokens_per_step(
        config,
        num_devices=num_devices,
    )

    old_total_tokens = int(
        config.runtime.total_tokens
    )

    new_total_tokens = int(
        max_steps
        * tokens_per_step
    )

    config.runtime.total_tokens = new_total_tokens

    print(
        "[train_tpu_optimized] max_steps override:\n"
        f"  max_steps={max_steps:,}\n"
        f"  tokens_per_step={tokens_per_step:,}\n"
        f"  runtime.total_tokens: {old_total_tokens:,} -> {new_total_tokens:,}",
        flush=True,
    )

    return config


def _report_backend_contract(config) -> None:
    """Make backend/sharding fields explicit for the active launcher."""
    backend = str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )
    sharding_strategy = str(config.optimizations.sharding_strategy)

    if backend == "pmap" and sharding_strategy != "pmap":
        print(
            "[train_tpu_optimized] warning: "
            f"optimizations.sharding_strategy={sharding_strategy!r} is not "
            "used by the PMAP trainer; runtime.backend='pmap' controls "
            "execution. Set sharding_strategy='pmap' to remove this warning.",
            flush=True,
        )


def _fresh_checkpoint_dir(config) -> None:
    ckpt_dir = Path(
        config.runtime.checkpoint_dir
    ).expanduser()

    if jax.process_index() != 0:
        return

    if ckpt_dir.exists():
        print(
            "[train_tpu_optimized] --fresh removing checkpoint_dir:\n"
            f"  {ckpt_dir}",
            flush=True,
        )

        shutil.rmtree(
            ckpt_dir
        )

    else:
        print(
            "[train_tpu_optimized] --fresh checkpoint_dir already clean:\n"
            f"  {ckpt_dir}",
            flush=True,
        )


def _apply_data_source_overrides(data_cfg, args) -> None:
    """Apply optional CLI dataset-source overrides without changing YAML defaults."""
    overrides = {
        "hf_repo_id": args.hf_repo_id,
        "hf_revision": args.hf_revision,
        "shard_directory": args.shard_directory,
        "shard_filename_prefix": args.shard_filename_prefix,
        "train_shard_start": args.train_shard_start,
        "train_shard_count": args.train_shard_count,
        "validation_shard_start": args.validation_shard_start,
        "validation_shard_count": args.validation_shard_count,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(data_cfg, field_name, value)

    if not data_cfg.hf_repo_id.strip():
        raise ValueError("data.hf_repo_id must be a non-empty dataset repository.")
    if not data_cfg.shard_directory.strip("/"):
        raise ValueError("data.shard_directory must be a non-empty folder path.")
    if not data_cfg.shard_filename_prefix.strip():
        raise ValueError("data.shard_filename_prefix must be non-empty.")
    if data_cfg.train_shard_count <= 0:
        raise ValueError("data.train_shard_count must be > 0.")
    if data_cfg.validation_shard_count < 0:
        raise ValueError("data.validation_shard_count must be >= 0.")

    # Hugging Face filenames are repository-relative, never absolute paths.
    data_cfg.shard_directory = data_cfg.shard_directory.strip("/")


def _write_run_manifest(
    *,
    args,
    config,
    train_paths,
    validation_paths,
    train_files,
    validation_files,
    num_devices: int,
    compilation_cache: dict,
) -> None:
    """Persist the resolved inputs needed to compare TPU runs later."""
    if jax.process_index() != 0:
        return

    package_names = (
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "orbax-checkpoint",
        "grain",
    )
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None

    repo_root = Path(__file__).resolve().parents[1]
    git_revision = None
    try:
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if git_result.returncode == 0:
            git_revision = git_result.stdout.strip()
    except OSError:
        pass

    manifest = {
        "manifest_version": 2,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "git_revision": git_revision,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "cli_args": vars(args),
        "resolved_config": config.model_dump(mode="json"),
        "jax": {
            "devices": [str(device) for device in jax.devices()],
            "local_device_count": int(num_devices),
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "attention_contract": {
            "variant": str(config.architecture.attention_variant),
            "implementation_requested": str(config.architecture.attention_impl),
            "fallback_policy": str(config.architecture.attention_fallback),
            "num_heads": int(config.model.num_heads),
            "num_kv_heads": (
                None
                if config.model.num_kv_heads is None
                else int(config.model.num_kv_heads)
            ),
            "splash_kv_expansion_expected": bool(
                config.architecture.attention_impl == "splash"
                and config.model.num_kv_heads is not None
                and config.model.num_kv_heads < config.model.num_heads
            ),
            "dispatch_confirmation": "startup/log evidence required",
        },
        "compilation_cache": compilation_cache,
        "data": {
            "train_local_paths": [str(path) for path in train_paths],
            "validation_local_paths": [
                str(path) for path in validation_paths
            ],
            "train_files": train_files,
            "validation_files": validation_files,
        },
    }

    manifest_path = (
        Path(config.runtime.checkpoint_dir).expanduser()
        / "run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        "[train_tpu_optimized] run manifest written:\n"
        f"  {manifest_path}",
        flush=True,
    )


def main():
    args = parse_args()

    print(
        f"JAX devices: {jax.devices()}",
        flush=True,
    )

    config = load_config(
        args.config,
        override_config=args.override_config,
    )

    data_cfg = config.data
    _apply_data_source_overrides(data_cfg, args)
    train_start = int(data_cfg.train_shard_start)
    train_count = int(data_cfg.train_shard_count)
    validation_count = int(data_cfg.validation_shard_count)
    validation_start = data_cfg.validation_shard_start

    if validation_count and validation_start is None:
        raise ValueError(
            "data.validation_shard_start is required when validation_shard_count > 0"
        )

    train_ids = list(range(train_start, train_start + train_count))
    validation_ids = (
        list(range(int(validation_start), int(validation_start) + validation_count))
        if validation_count
        else []
    )
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(
            f"Train/validation shard overlap is not allowed: {sorted(overlap)}"
        )

    def _shard_name(shard_id: int) -> str:
        return (
            f"{data_cfg.shard_directory}/"
            f"{data_cfg.shard_filename_prefix}_{shard_id:05d}.bin"
        )

    train_files = [_shard_name(shard_id) for shard_id in train_ids]
    validation_files = [
        _shard_name(shard_id) for shard_id in validation_ids
    ]

    cache_dir = data_cfg.hf_cache_dir
    download_kwargs = {
        "repo_id": data_cfg.hf_repo_id,
        "repo_type": "dataset",
    }
    resolved_revision = data_cfg.hf_revision or "default branch"
    print(
        "[data] Hugging Face dataset source: "
        f"{data_cfg.hf_repo_id} @ {resolved_revision}",
        flush=True,
    )
    print(
        "[data] shard selector: "
        f"folder={data_cfg.shard_directory}, "
        f"prefix={data_cfg.shard_filename_prefix}, "
        f"train={train_start}:{train_start + train_count - 1}",
        flush=True,
    )
    if data_cfg.hf_revision:
        download_kwargs["revision"] = data_cfg.hf_revision
    if cache_dir:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        download_kwargs["cache_dir"] = str(cache_path)
        print(
            f"[data] Hugging Face cache directory: {cache_path}",
            flush=True,
        )

    def _download(files, label: str):
        print(
            f"[data] downloading {label} shards: {len(files)}",
            flush=True,
        )
        for filename in files:
            print(f"  {filename}", flush=True)
        return [
            hf_hub_download(filename=filename, **download_kwargs)
            for filename in files
        ]

    paths = _download(train_files, "train")
    validation_paths = (
        _download(validation_files, "validation")
        if validation_files
        else []
    )

    compilation_cache = _compilation_cache_snapshot(config)
    _configure_persistent_compilation_cache(config)

    # Single-VM TPU mode relies on JAX runtime discovery; do not call jax.distributed.initialize().
    num_devices = jax.local_device_count()

    config = _apply_max_steps_override(
        config,
        max_steps=args.max_steps,
        num_devices=num_devices,
    )

    if args.fresh:
        _fresh_checkpoint_dir(
            config
        )

    _report_backend_contract(config)

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * num_devices
    )

    tokens_per_step = _tokens_per_step(
        config,
        num_devices=num_devices,
    )

    total_steps = (
        int(config.runtime.total_tokens)
        // tokens_per_step
    )

    print(
        "[train_tpu_optimized] config:\n"
        f"  path={args.config}\n"
        f"  checkpoint_dir={config.runtime.checkpoint_dir}\n"
        f"  global_batch_size={global_batch_size}\n"
        f"  gradient_accumulation={config.runtime.gradient_accumulation}\n"
        f"  prefetch_size={config.runtime.prefetch_size}\n"
        f"  num_heads={config.model.num_heads}\n"
        f"  num_kv_heads={config.model.num_kv_heads}\n"
        f"  sharding_strategy={config.optimizations.sharding_strategy}\n"
        f"  param_dtype={config.spmd.dtype.param_dtype}\n"
        f"  compute_dtype={config.spmd.dtype.compute_dtype}\n"
        f"  output_dtype={config.spmd.dtype.output_dtype}\n"
        f"  remat_policy={config.spmd.remat.policy}\n"
        f"  remat_granularity={config.spmd.remat.granularity}\n"
        f"  scan_layers={config.spmd.remat.scan_layers}\n"
        f"  prevent_cse={config.spmd.remat.prevent_cse}\n"
        f"  attention_variant={config.architecture.attention_variant}\n"
        f"  attention_impl={config.architecture.attention_impl}\n"
        f"  chunked_logits={config.loss.chunked_logits}\n"
        f"  logits_chunk_size={config.loss.logits_chunk_size}\n"
        f"  loss_backend={config.loss.backend}\n"
        f"  tokamax_implementation={config.loss.tokamax_implementation}\n"
        f"  compilation_cache_dir={config.optimizations.compilation_cache_dir}\n"
        f"  tokens_per_step={tokens_per_step:,}\n"
        f"  total_tokens={int(config.runtime.total_tokens):,}\n"
        f"  total_steps={total_steps:,}",
        flush=True,
    )

    _write_run_manifest(
        args=args,
        config=config,
        train_paths=paths,
        validation_paths=validation_paths,
        train_files=train_files,
        validation_files=validation_files,
        num_devices=num_devices,
        compilation_cache=compilation_cache,
    )

    # --- UPDATED DATA LOADER CREATION ---
    dataset = create_dataloader(
        config=config,
        paths=paths,
        global_batch_size=global_batch_size,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )
    # ------------------------------------

    trainer = Trainer(
        config=config,
        resume_dir=config.runtime.checkpoint_dir,
    )

    eval_dataset = (
        create_dataloader(
            config=config,
            paths=validation_paths,
            global_batch_size=global_batch_size,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
        )
        if validation_paths
        else None
    )

    trainer.train(
        dataset,
        eval_dataloader=eval_dataset,
    )


if __name__ == "__main__":
    main()
