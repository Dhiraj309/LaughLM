
from typing import List
from tokenizers import Tokenizer

class LaughTokenizer:
    """
    Wrapper around HuggingFace fast tokenizer.

    Uses Rust-backed tokenizer for high throughput.
    """

    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.eos_id = self.tokenizer.token_to_id("<eos>")

        if self.eos_id is None:
            raise ValueError("Tokenizer must contain <eos> token")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(texts).ids

    def encode_batch(self, text: List[str]) -> List[List[int]]:
        encodings = self.tokenizer.encode_batch(text)

        return [e.ids for e in encodings]

    def add_eos(self, tokens: List[int]) ->List[int]:
        return tokens + [self.eos_id]
