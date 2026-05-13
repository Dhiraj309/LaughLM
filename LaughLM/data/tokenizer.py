from typing import List
from tokenizers import Tokenizer


class LaughTokenizer:
    """
    LaughLM tokenizer wrapper around HuggingFace tokenizers.

    Provides a clean interface for encoding/decoding with proper
    special token handling.

    FIX (frontier-optim audit 2026):
      Added missing add_eos() method that shard_writer.py calls.
      Without this, BinaryShardWriter.add_document() would crash
      with AttributeError.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (no special tokens)."""
        return self.tokenizer.encode(text).ids

    def encode_with_special(self, text: str) -> List[int]:
        """Encode text with BOS and EOS tokens."""
        return [self.bos_id] + self.encode(text) + [self.eos_id]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encode with BOS and EOS tokens."""
        encodings = self.tokenizer.encode_batch(texts)
        return [[self.bos_id] + e.ids + [self.eos_id] for e in encodings]

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def add_eos(self, tokens: List[int]) -> List[int]:
        """Append EOS token to a token sequence.
        
        Used by shard_writer.py to mark document boundaries in
        pre-tokenized training shards. The EOS token allows the
        model to learn document-end prediction.
        """
        return tokens + [self.eos_id]
