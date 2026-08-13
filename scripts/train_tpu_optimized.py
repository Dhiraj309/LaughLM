from __future__ import annotations

import argparse
import os
import shutil
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


def main():
    args = parse_args()

    print(
        f"JAX devices: {jax.devices()}",
        flush=True,
    )

    files = [
        f"fineweb_edu_100bt/fineweb_edu_100bt_shard_{i:05d}.bin"
        for i in range(0,1)
    ]

    print(
        "Downloading shards:",
        flush=True,
    )

    for f in files:
        print(
            f"  {f}",
            flush=True,
        )

    paths = [
        hf_hub_download(
            repo_id="LaughTaleAI/LaughLM-Tokenized-Fine",
            filename=f,
            repo_type="dataset",
        )
        for f in files
    ]

    config = load_config(
        args.config
    )

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
        f"  loss_backend={config.loss.backend}\n"
        f"  tokamax_implementation={config.loss.tokamax_implementation}\n"
        f"  tokens_per_step={tokens_per_step:,}\n"
        f"  total_tokens={int(config.runtime.total_tokens):,}\n"
        f"  total_steps={total_steps:,}",
        flush=True,
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

    trainer.train(
        dataset
    )


if __name__ == "__main__":
    main()
