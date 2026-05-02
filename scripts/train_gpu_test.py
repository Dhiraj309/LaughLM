"""
scripts/train_gpu_test.py

GPU test training script for LaughLM.
Downloads a pre-tokenized shard and runs training with the gpu_test config.
"""

from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset

import jax
jax.config.update("jax_default_matmul_precision", "high")


def main():

    # ── Download dataset shard ────────────────────────────────
    path = hf_hub_download(
        repo_id="LaughTaleAI/fineweb-edu-gpt2-tokenized",
        filename="train_00000.bin",
        repo_type="dataset",
    )

    # ── Load configuration ────────────────────────────────────
    config = load_config("configs/gpu_test.yaml")

    # ── Dataset ───────────────────────────────────────────────
    # batch_size = GLOBAL = micro_batch_per_device × num_devices
    # Gradient accumulation is handled inside the Trainer.
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
