from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.training.fsdp_trainer import FSDPTrainer
from LaughLM.data.memmap_loader import MemmapDataset


DEFAULT_CONFIG = (
    "configs/v5e_fsdp_proxy_d4_f2_fusedqkv_benchmark.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LaughLM with FSDP."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to FSDP config YAML.",
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


def _tokens_per_step(config, *, data_replicas: int) -> int:
    return int(
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * data_replicas
        * config.runtime.gradient_accumulation
    )


def _apply_max_steps_override(
    config,
    *,
    max_steps: int | None,
    data_replicas: int,
):
    if max_steps is None:
        return config

    if max_steps <= 0:
        raise ValueError(
            "--max_steps must be > 0"
        )

    tokens_per_step = _tokens_per_step(
        config,
        data_replicas=data_replicas,
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
        "[train_tpu_fsdp] max_steps override:\n"
        f"  max_steps={max_steps:,}\n"
        f"  tokens_per_step={tokens_per_step:,}\n"
        f"  runtime.total_tokens: {old_total_tokens:,} -> {new_total_tokens:,}\n"
        f"  scheduler.horizon_tokens remains={config.scheduler.horizon_tokens:,}",
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
            "[train_tpu_fsdp] --fresh removing checkpoint_dir:\n"
            f"  {ckpt_dir}",
            flush=True,
        )

        shutil.rmtree(
            ckpt_dir
        )

    else:
        print(
            "[train_tpu_fsdp] --fresh checkpoint_dir already clean:\n"
            f"  {ckpt_dir}",
            flush=True,
        )


def main():
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")

    repo_id = "LaughTaleAI/LaughLM-Tokenized-Fine"
    folder = "LaughLM-v0.2-cpt-smollm-edu-1B"

    files = [
        f"{folder}/LaughLM-v0.2-cpt-smollm-edu-1B_shard_{i:05d}.bin"
        for i in range(0, 10)
    ]

    print("Downloading CPT shards:")
    for f in files:
        print(" ", f)

    paths = [
        hf_hub_download(
            repo_id=repo_id,
            filename=f,
            repo_type="dataset",
        )
        for f in files
    ]

    config = load_config(
        args.config
    )

    data_replicas = config.spmd.mesh.axis_sizes()["data"]

    config = _apply_max_steps_override(
        config,
        max_steps=args.max_steps,
        data_replicas=data_replicas,
    )

    if args.fresh:
        _fresh_checkpoint_dir(
            config
        )

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * data_replicas
    )

    tokens_per_step = _tokens_per_step(
        config,
        data_replicas=data_replicas,
    )

    total_steps = (
        int(config.runtime.total_tokens)
        // tokens_per_step
    )

    print(
        "[train_tpu_fsdp] config:\n"
        f"  path={args.config}\n"
        f"  checkpoint_dir={config.runtime.checkpoint_dir}\n"
        f"  global_batch_size={global_batch_size}\n"
        f"  gradient_accumulation={config.runtime.gradient_accumulation}\n"
        f"  tokens_per_step={tokens_per_step:,}\n"
        f"  total_tokens={int(config.runtime.total_tokens):,}\n"
        f"  total_steps={total_steps:,}",
        flush=True,
    )

    dataset = MemmapDataset(
        paths=paths,
        seq_len=config.runtime.seq_len,
        global_batch_size=global_batch_size,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    trainer = FSDPTrainer(
        config=config,
        resume_dir=config.runtime.checkpoint_dir,
    )

    trainer.train(
        dataset
    )


if __name__ == "__main__":
    main()
