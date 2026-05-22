from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config

from LaughLM.training.trainer import (
    Trainer,
)

from LaughLM.data.memmap_loader import (
    MemmapDataset,
)

from LaughLM.utils.memory import (
    print_memory_stats,
)

import jax


def main():

    # ========================================================
    # JAX runtime
    # ========================================================

    print(
        f"JAX devices: "
        f"{jax.devices()}",
        flush=True,
    )

    print(
        f"Device count: "
        f"{jax.device_count()}",
        flush=True,
    )

    print(
        f"Process count: "
        f"{jax.process_count()}",
        flush=True,
    )

    # ========================================================
    # Memory stats
    # ========================================================

    print_memory_stats(
        prefix="[startup] ",
    )

    # ========================================================
    # Dataset shards
    # ========================================================

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

    # ========================================================
    # Config
    # ========================================================

    config = load_config(
        "configs/v5e_smoke.yaml"
    )

    # ========================================================
    # Dataset
    # ========================================================

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * jax.device_count()
    )

    dataset = MemmapDataset(
        paths=paths,

        seq_len=(
            config.runtime.seq_len
        ),

        global_batch_size=(
            global_batch_size
        ),

        process_index=(
            jax.process_index()
        ),

        process_count=(
            jax.process_count()
        ),
    )

    # ========================================================
    # Trainer
    # ========================================================

    print(
        "[main] building trainer...",
        flush=True,
    )

    trainer = Trainer(
        config=config,
        resume_dir=(
            config.runtime
            .checkpoint_dir
        ),
    )

    print_memory_stats(
        prefix="[post-init] ",
    )

    print(
        "[main] trainer initialized",
        flush=True,
    )

    # ========================================================
    # Train
    # ========================================================

    trainer.train(dataset)

    # ========================================================
    # Final memory stats
    # ========================================================

    print_memory_stats(
        prefix="[shutdown] ",
    )


if __name__ == "__main__":

    main()
