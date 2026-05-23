from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.training.fsdp_trainer import FSDPTrainer
from LaughLM.data.memmap_loader import MemmapDataset


def main():
    print(f"JAX devices: {jax.devices()}")

    files = [
        "fineweb-edu/fineweb-edu_shard_00008.bin",
        "fineweb-edu/fineweb-edu_shard_00009.bin",
        "fineweb-edu/fineweb-edu_shard_00010.bin",
        "fineweb-edu/fineweb-edu_shard_00011.bin",
    ]

    paths = [
        hf_hub_download(
            repo_id="LaughTaleAI/LaughLM-Tokenized",
            filename=f,
            repo_type="dataset",
        )
        for f in files
    ]

    config = load_config("configs/v5e_fsdp_smoke.yaml")

    data_replicas = config.spmd.mesh.axis_sizes()["data"]

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

    trainer.train(dataset)


if __name__ == "__main__":
    main()