
from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config

from LaughLM.training.trainer import (
    Trainer,
)

from LaughLM.data.memmap_loader import (
    MemmapDataset,
)

import jax


def main():

    print(
        f"JAX devices: "
        f"{jax.devices()}"
    )

    files = [
        "fineweb-edu/fineweb-edu_shard_00012.bin",
    ]

    paths = [

        hf_hub_download(
            repo_id="LaughTaleAI/LaughLM-Tokenized",
            filename=f,
            repo_type="dataset",
        )

        for f in files
    ]

    config = load_config(
        "configs/v5e_smoke.yaml"
    )

    dataset = MemmapDataset(
        paths=paths,

        seq_len=(
            config.runtime.seq_len
        ),

        batch_size=(
            config.runtime
            .micro_batch_per_device
            * jax.device_count()
        ),

        process_index=(
            jax.process_index()
        ),

        process_count=(
            jax.process_count()
        ),
    )

    trainer = Trainer(
        config=config,
        resume_dir=(
            config.runtime
            .checkpoint_dir
        ),
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
