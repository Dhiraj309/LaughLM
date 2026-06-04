from huggingface_hub import hf_hub_download
import jax

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset


def main():
    print(f"JAX devices: {jax.devices()}")

    repo_id = "LaughTaleAI/LaughLM-Tokenized-Fine"
    folder = "LaughLM-v0.2-cpt-smollm-edu-1B"

    # Use only full 10M-token shards.
    # Skip shard_00010 because it is the tiny final partial shard.
    files = [
        f"{folder}/LaughLM-v0.2-cpt-smollm-edu-1B_shard_{i:05d}.bin"
        for i in range(50, 100)
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

    config = load_config("configs/v5e_pmap_cpt_smoke_100m.yaml")

    global_batch_size = (
        config.runtime.micro_batch_per_device
        * jax.local_device_count()
    )

    print("global_batch_size before grad accum:", global_batch_size)
    print("seq_len:", config.runtime.seq_len)
    print("grad_accum:", config.runtime.gradient_accumulation)

    tokens_per_optimizer_step = (
        global_batch_size
        * config.runtime.seq_len
        * config.runtime.gradient_accumulation
    )

    print("tokens_per_optimizer_step:", tokens_per_optimizer_step)

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
