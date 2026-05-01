import numpy as np
from typing import Iterator, Tuple

from LaughLM.data.tokenizer import LaughTokenizer


class BinaryShardWriter:
    """
    Writes pretokenized training shards.

    Output format
    -------------
    uint16 flat token stream.

    Each shard contains:
        shard_tokens tokens
    """

    def __init__(
        self,
        tokenizer: LaughTokenizer,
        output_path: str,
        shard_tokens: int,
        flush_tokens: int = 1_000_000,
    ):

        self.tokenizer = tokenizer
        self.output_path = output_path
        self.shard_tokens = shard_tokens
        self.flush_tokens = flush_tokens

        self.buffer = []
        self.total_written = 0

    # ------------------------------------------------------------
    # Add document
    # ------------------------------------------------------------

    def add_document(self, text: str):

        tokens = self.tokenizer.encode(text)

        tokens = self.tokenizer.add_eos(tokens)

        # Safety check for uint16
        if tokens and max(tokens) >= 65535:
            raise ValueError("Token id exceeds uint16 range")

        self.buffer.extend(tokens)

    # ------------------------------------------------------------
    # Flush buffer to disk
    # ------------------------------------------------------------

    def flush(self):

        if not self.buffer:
            return

        arr = np.array(self.buffer, dtype=np.uint16)

        with open(self.output_path, "ab") as f:
            arr.tofile(f)

        self.total_written += len(arr)

        self.buffer = []

    # ------------------------------------------------------------
    # Build shard
    # ------------------------------------------------------------

    def build_shard(self, sampler: Iterator[Tuple[str, str]]):

        print(f"\nBuilding shard → {self.output_path}")

        while self.total_written < self.shard_tokens:

            text, _ = next(sampler)

            self.add_document(text)

            if len(self.buffer) >= self.flush_tokens:
                self.flush()

            if (
                self.total_written > 0
                and self.total_written % 100_000_000 == 0
            ):
                print(f"{self.total_written:,} tokens written")

        self.flush()

        print(f"Shard complete: {self.total_written:,} tokens")
