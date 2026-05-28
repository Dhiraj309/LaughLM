from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset


def main():
    print(f"JAX devices: {jax.devices()}")

    files = [
        # "fineweb-edu/fineweb-edu_shard_00000.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00001.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00002.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00003.bin", # Done

        # "fineweb-edu/fineweb-edu_shard_00004.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00005.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00006.bin", # Done
        # "fineweb-edu/fineweb-edu_shard_00007.bin", # Done

        "fineweb-edu/fineweb-edu_shard_00008.bin", # Done
        "fineweb-edu/fineweb-edu_shard_00009.bin", # Done
        "fineweb-edu/fineweb-edu_shard_00010.bin", # Done
        "fineweb-edu/fineweb-edu_shard_00011.bin", # Done

        # "fineweb-edu/fineweb-edu_shard_00012.bin",
        # "fineweb-edu/fineweb-edu_shard_00013.bin",
        # "fineweb-edu/fineweb-edu_shard_00014.bin",
        # "fineweb-edu/fineweb-edu_shard_00015.bin",

    ]

    paths = [
        hf_hub_download(
            repo_id="LaughTaleAI/LaughLM-Tokenized-Fine",
            filename=f,
            repo_type="dataset",
        )
        for f in files
    ]

    config = load_config("configs/v5e_pmap.yaml")

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * jax.local_device_count()
    )

    dataset = MemmapDataset(
        paths=paths,
        seq_len=config.runtime.seq_len,
        global_batch_size=global_batch_size,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )

    trainer = Trainer(
        config=config,
        resume_dir=config.runtime.checkpoint_dir,
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
