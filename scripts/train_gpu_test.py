
from huggingface_hub import hf_hub_download

from LaughLM.config.loader import load_config
from LaughLM.training.trainer import Trainer
from LaughLM.data.memmap_loader import MemmapDataset

def main():

    # download shard
    path = hf_hub_download(
        repo_id="LaughTaleAI/fineweb-edu-gpt2-tokenized",
        filename="train_00000.bin",
        repo_type="dataset"
    )

    config = load_config("configs/gpu_test.yaml")

    dataset = MemmapDataset(
        paths=path,
        seq_len=config.runtime.seq_len,
        batch_size=config.runtime.micro_batch_per_device
    )

    trainer = Trainer(config)

    trainer.train(dataset)


if __name__ == "__main__":
    main()
