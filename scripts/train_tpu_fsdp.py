from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.training.fsdp_trainer import FSDPTrainer
from LaughLM.data.memmap_loader import MemmapDataset


def _canonical_backend(config) -> str:
    return str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )


def _validate_fsdp_config(config, config_path: str) -> None:
    backend = _canonical_backend(
        config
    )

    if backend != "fsdp":
        raise ValueError(
            "scripts/train_tpu_fsdp.py only supports FSDP configs.\n"
            f"  config:            {config_path}\n"
            f"  raw backend:       {config.runtime.backend!r}\n"
            f"  canonical backend: {backend!r}\n"
            "Use scripts/train.py for backend registry dispatch, or use "
            "a PMAP-specific launcher for runtime.backend='pmap'."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v5e_fsdp_smoke.yaml",
        help="FSDP config YAML path.",
    )

    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    config = load_config(
        args.config
    )

    _validate_fsdp_config(
        config,
        args.config,
    )

    repo_id = "LaughTaleAI/LaughLM-Tokenized-Fine"
    folder = "LaughLM-v0.2-cpt-smollm-edu-1B"

    # Use only full 10M-token shards.
    # Skip shard_00010 because it is the tiny final partial shard.
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

    data_replicas = int(
        config.spmd.mesh.axis_sizes()["data"]
    )

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * data_replicas
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
