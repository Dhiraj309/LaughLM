from typing import List
from tokenizers import Tokenizer


class LaughTokenizer:
    """
    Wrapper around HuggingFace fast tokenizer.

    Uses Rust-backed tokenizer for high throughput.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.eos_id = self.tokenizer.token_to_id("<eos>")

        if self.eos_id is None:
            raise ValueError(
                "Tokenizer must contain <eos> token. "
                "Ensure your tokenizer was trained with special_tokens=['<eos>']."
            )

        self.pad_id = self.tokenizer.token_to_id("<pad>") or 0

        self.vocab_size = self.tokenizer.get_vocab_size()

    # ------------------------------------------------------------------
    # Single document encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Encode a single string to token IDs.
        Does NOT add EOS — call add_eos() separately if needed.
        """
        # FIX: was 'texts' (typo) — now correctly uses 'text'
        return self.tokenizer.encode(text).ids

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a list of strings. Uses Rust parallelism internally.
        """
        encodings = self.tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    # ------------------------------------------------------------------
    # EOS handling
    # ------------------------------------------------------------------

    def add_eos(self, tokens: List[int]) -> List[int]:
        """
        Append EOS token to a token sequence.
        Called by ShardWriter after each document.
        """
        return tokens + [self.eos_id]

    # ------------------------------------------------------------------
    # Decode (for generation / debugging)
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
