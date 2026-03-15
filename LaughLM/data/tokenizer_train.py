
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

from LaughLM.data.domain_sampler import DomainSampler

def train_tokenizer(
        sources,
        vocab_size=32000,
        output_path="tokenizer.json",
        max_samples=5_000_000,
    ):
    """
    Train a BPE tokenizer from streaming datasets.
    """

    print("Initalizing Tokenizer...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = NFKC()

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
        ],
    )

    print("Streaming Dataset...")
    sampler=DomainSampler(sources)

    def text_iterator():
        for i, (text, _) in enumerate(sampler):
            if i >= max_samples:
                break

            yield text

            if i % 10_000 == 0:
                print(f"Processed {i:,} samples")

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
    )

    tokenizer.post_processor = TemplateProcessing(
        single="$A <eos>",
        special_tokens=[
            ("<eos>", tokenizer.token_to_id("<eos>"))
        ],
    )

    print(f"Saving tokenizer to {output_path}")
    tokenizer.save(output_path)

    print("Tokenizer Training Complete")
