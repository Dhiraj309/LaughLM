
from LaughLM.data.dataset import DomainSampler
from LaughLM.data.tokenizer import LaughTokenizer
from LaughLM.data.shard_writer import BinaryShardWriter


def main():

    tokenizer = LaughTokenizer("tokenizer.json")

    sampler = DomainSampler(
        sources=[
            {"name": "HuggingFaceFW/fineweb-edu", "weight": 0.4},
            {"name": "bigcode/starcoderdata", "weight": 0.2},
        ]
    )

    writer = BinaryShardWriter(
        tokenizer=tokenizer,
        output_path="train_000.bin",
        shard_tokens=5_000_000_000,
    )

    writer.build_shard(iter(sampler))


if __name__ == "__main__":
    main()
