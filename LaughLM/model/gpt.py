from flax import linen as nn
import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.transformer_block import TransformerBlock
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.positional import build_positional_encoding


class GPTModel(nn.Module):
    """
    Decoder-only GPT architecture.
    """

    config: LaughLMConfig

    def setup(self):

        d_model = self.config.model.d_model
        vocab_size = self.config.model.vocab_size
        num_layers = self.config.model.num_layers

        # token embedding
        self.token_embedding = nn.Embed(
            num_embeddings=vocab_size,
            features=d_model
        )

        # positional encoding
        self.positional = build_positional_encoding(self.config)

        # transformer stack
        self.blocks = [
            TransformerBlock(config=self.config)
            for _ in range(num_layers)
        ]

        # final normalization
        self.final_norm = build_normalization(self.config)

        # output projection
        if not self.config.architecture.weight_tying:
            self.lm_head = nn.Dense(vocab_size)

    def __call__(self, input_ids):

        seq_len = input_ids.shape[1]

        # token embeddings
        x = self.token_embedding(input_ids)

        # positional encoding
        if self.positional is not None:

            positions = jnp.arange(seq_len)[None, :]

            pos_emb = self.positional(positions)

            x = x + pos_emb

        # transformer layers
        for block in self.blocks:
            x = block(x)

        # final norm
        x = self.final_norm(x)

        # output logits
        if self.config.architecture.weight_tying:

            logits = jnp.einsum(
                "btd,vd->btv",
                x,
                self.token_embedding.embedding
            )

        else:
            logits = self.lm_head(x)

        return logits
