from typing import List
from tokenizers import Tokenizer


class LaughTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def encode_with_special(self, text: str) -> List[int]:
        return [self.bos_id] + self.encode(text) + [self.eos_id]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        encodings = self.tokenizer.encode_batch(texts)
        return [[self.bos_id] + e.ids + [self.eos_id] for e in encodings]

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
