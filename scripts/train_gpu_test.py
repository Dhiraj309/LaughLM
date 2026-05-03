
from huggingface_hub import hf_hub_download
from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset
import jax

jax.config.update("jax_default_matmul_precision", "high")


def main():
    # ------------------------------------------------------------
    # Download dataset shard
    # ------------------------------------------------------------
    path = hf_hub_download(
        repo_id="LaughTaleAI/fineweb-edu-gpt2-tokenized",
        filename="train_00000.bin",
        repo_type="dataset",
    )

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------
    config = load_config("configs/gpu_test.yaml")

    # ------------------------------------------------------------
    # Dataset
    # IMPORTANT:
    # The dataset should produce MICRO batches.
    # Gradient accumulation is handled inside the Trainer.
    # ------------------------------------------------------------
    dataset = MemmapDataset(
        paths=path,
        seq_len=config.runtime.seq_len,
        batch_size=config.runtime.micro_batch_per_device,
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer = Trainer(config)

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    trainer.train(dataset)


if __name__ == "__main__":
    main()
