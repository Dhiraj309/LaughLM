# tests/test_kv_cache_parity.py

import numpy as np

import jax
import jax.numpy as jnp

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.model.llama.kv_cache import (
    init_kv_cache,
)


def build_test_config():

    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def create_kv_caches(
    config,
    batch_size,
):

    return [
        init_kv_cache(
            batch_size=batch_size,
            max_seq_len=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.compute_dtype,
        )
        for _ in range(
            config.num_hidden_layers
        )
    ]


def test_kv_cache_decode_parity():

    config = build_test_config()

    model = LlamaForCausalLM(
        config=config
    )

    rng = jax.random.PRNGKey(0)

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5]],
        dtype=jnp.int32,
    )

    variables = model.init(
        rng,
        input_ids=input_ids,
        use_cache=False,
        mode="train",
    )

    # --------------------------------------------------
    # Full forward
    # --------------------------------------------------

    full_logits, _ = model.apply(
        variables,
        input_ids=input_ids,
        use_cache=False,
        mode="train",
    )

    full_last_logits = (
        full_logits[:, -1, :]
    )

    # --------------------------------------------------
    # Prefill
    # --------------------------------------------------

    prefill_ids = input_ids[:, :-1]

    kv_caches = create_kv_caches(
        config,
        batch_size=1,
    )

    _, kv_caches = model.apply(
        variables,
        input_ids=prefill_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="prefill",
    )

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    decode_ids = input_ids[:, -1:]

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

    np.testing.assert_allclose(
        np.asarray(full_last_logits),
        np.asarray(decode_last_logits),
        rtol=1e-5,
        atol=1e-5,
    )


def test_iterative_decode_parity():

    config = build_test_config()

    model = LlamaForCausalLM(
        config=config
    )

    rng = jax.random.PRNGKey(42)

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7]],
        dtype=jnp.int32,
    )

    variables = model.init(
        rng,
        input_ids=input_ids,
        use_cache=False,
        mode="train",
    )

    # --------------------------------------------------
    # Full forward reference
    # --------------------------------------------------

    full_logits, _ = model.apply(
        variables,
        input_ids=input_ids,
        use_cache=False,
        mode="train",
    )

    # --------------------------------------------------
    # Prefill
    # --------------------------------------------------

    prefill_ids = input_ids[:, :4]

    kv_caches = create_kv_caches(
        config,
        batch_size=1,
    )

    _, kv_caches = model.apply(
        variables,
        input_ids=prefill_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="prefill",
    )

    # --------------------------------------------------
    # Iterative decode
    # --------------------------------------------------

    for step_idx in range(4, 7):

        decode_ids = input_ids[
            :,
            step_idx:step_idx + 1,
        ]

        decode_logits, kv_caches = (
            model.apply(
                variables,
                input_ids=decode_ids,
                kv_caches=kv_caches,
                use_cache=True,
                mode="decode",
            )
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