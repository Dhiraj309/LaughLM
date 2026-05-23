from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.data.memmap_loader import MemmapDataset
from LaughLM.training.trainer import Trainer
from LaughLM.training.fsdp_trainer import FSDPTrainer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    config = load_config(args.config)

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

    if config.runtime.backend == "gspmd":
        data_replicas = config.spmd.mesh.axis_sizes()["data"]
        trainer_cls = FSDPTrainer
    elif config.runtime.backend == "pmap":
        data_replicas = jax.device_count()
        trainer_cls = Trainer
    else:
        raise ValueError(f"Unknown runtime.backend: {config.runtime.backend}")

    dataset = MemmapDataset(
        paths=paths,
        seq_len=config.runtime.seq_len,
        global_batch_size=(
            config.runtime.micro_batch_per_device
            * data_replicas
        ),
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    trainer = trainer_cls(
        config=config,
        resume_dir=config.runtime.checkpoint_dir,
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()