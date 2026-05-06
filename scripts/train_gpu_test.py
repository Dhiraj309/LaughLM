"""
scripts/train_gpu_test.py

Training script for LaughLM.
Downloads pre-tokenized shards and runs training.

NOTE: Do NOT set jax_default_matmul_precision='high' for TPU.
It forces f32 accumulation which halves MXU throughput.
bf16 native precision is correct for TPU training.
"""

from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset

import jax


def main():

    print(f"JAX devices: {jax.devices()}")

    # ── Download dataset shard ────────────────────────────────
    path = hf_hub_download(
        repo_id="LaughTaleAI/fineweb-edu-gpt2-tokenized",
        filename="train_00000.bin",
        repo_type="dataset",
    )

    # ── Load configuration ────────────────────────────────────
    config = load_config("configs/gpu_test.yaml")

    # ── Dataset ───────────────────────────────────────────────
    num_devices = jax.device_count()

    dataset = MemmapDataset(
        paths=path,
        seq_len=config.runtime.seq_len,
        batch_size=config.runtime.micro_batch_per_device * num_devices,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    # ── Train ─────────────────────────────────────────────────
    trainer = Trainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()