"""
tests/test_kv_cache_parity.py

Critical architecture correctness test.

Invariant:
    Full forward logits == prefill + decode logits

This is the single most important correctness test
for autoregressive transformer inference.

The test validates:
- RoPE position semantics
- KV cache append semantics
- causal masking correctness
- decode-time attention correctness
- GQA cache behavior
- deterministic cache indexing

Expected cache layout:
    key/value:
        [B, S, KVH, Dh]

Expected attention layout:
    query:
        [B, QH, T, Dh]
"""

import numpy as np

import jax
import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.model.llama.kv_cache import init_kv_cache


def build_test_config() -> LlamaConfig:
    """
    Tiny deterministic config for parity testing.
    """

    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
    )


def test_kv_cache_decode_parity():

    config = build_test_config()

    model = LlamaForCausalLM(config)

    rng = jax.random.PRNGKey(0)

    # ---------------------------------------------------------
    # Input sequence
    # ---------------------------------------------------------

    #
    # Sequence:
    #
    # [1, 2, 3, 4, 5]
    #
    # We compare:
    #
    # full([1 2 3 4 5])
    #
    # vs
    #
    # prefill([1 2 3 4])
    # decode([5])
    #

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5]],
        dtype=jnp.int32,
    )

    # ---------------------------------------------------------
    # Initialize parameters
    # ---------------------------------------------------------

    variables = model.init(
        rng,
        input_ids=input_ids,
        use_cache=False,
    )

    # ---------------------------------------------------------
    # Full forward pass
    # ---------------------------------------------------------

    full_logits, _ = model.apply(
        variables,
        input_ids=input_ids,
        use_cache=False,
    )

    #
    # Compare final-token logits only
    #

    full_last_logits = full_logits[:, -1, :]

    # ---------------------------------------------------------
    # Prefill
    # ---------------------------------------------------------

    prefill_ids = input_ids[:, :-1]

    batch_size = prefill_ids.shape[0]

    kv_caches = []

    for _ in range(config.num_hidden_layers):

        kv_caches.append(
            init_kv_cache(
                batch_size=batch_size,
                max_seq_len=config.max_position_embeddings,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=jnp.float32,
            )
        )

    prefill_logits, kv_caches = model.apply(
        variables,
        input_ids=prefill_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="prefill",
    )

    # ---------------------------------------------------------
    # Single-token decode
    # ---------------------------------------------------------

    decode_ids = input_ids[:, -1:]

    decode_logits, kv_caches = model.apply(
        variables,
        input_ids=decode_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="decode",
    )

    decode_last_logits = decode_logits[:, -1, :]

    # ---------------------------------------------------------
    # Numerical parity assertion
    # ---------------------------------------------------------

    np.testing.assert_allclose(
        np.asarray(full_last_logits),
        np.asarray(decode_last_logits),
        rtol=1e-5,
        atol=1e-5,
    )

def test_iterative_decode_parity():

    config = build_test_config()

    model = LlamaForCausalLM(config)

    rng = jax.random.PRNGKey(42)

    #
    # Full sequence
    #
    # [1,2,3,4,5,6,7]
    #

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7]],
        dtype=jnp.int32,
    )

    variables = model.init(
        rng,
        input_ids=input_ids,
        use_cache=False,
    )

    # ---------------------------------------------------------
    # Full forward reference
    # ---------------------------------------------------------

    full_logits, _ = model.apply(
        variables,
        input_ids=input_ids,
        use_cache=False,
    )

    # ---------------------------------------------------------
    # Prefill
    # ---------------------------------------------------------

    prefill_ids = input_ids[:, :4]

    batch_size = prefill_ids.shape[0]

    kv_caches = []

    for _ in range(config.num_hidden_layers):

        kv_caches.append(
            init_kv_cache(
                batch_size=batch_size,
                max_seq_len=config.max_position_embeddings,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=jnp.float32,
            )
        )

    _, kv_caches = model.apply(
        variables,
        input_ids=prefill_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="prefill",
    )

    # ---------------------------------------------------------
    # Iterative decode
    # ---------------------------------------------------------

    for step_idx in range(4, 7):

        decode_ids = input_ids[
            :,
            step_idx:step_idx + 1,
        ]

        decode_logits, kv_caches = model.apply(
            variables,
            input_ids=decode_ids,
            kv_caches=kv_caches,
            use_cache=True,
            mode="decode",
        )

        decode_last_logits = (
            decode_logits[:, -1, :]
        )

        reference_logits = (
            full_logits[:, step_idx, :]
        )

        np.testing.assert_allclose(
            np.asarray(reference_logits),
            np.asarray(decode_last_logits),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":

    test_kv_cache_decode_parity()

    print("\nKV-cache parity test passed.\n")