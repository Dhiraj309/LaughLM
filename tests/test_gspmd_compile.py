# tests/test_gspmd_compile.py

from __future__ import annotations

import jax
import jax.numpy as jnp

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.training.loss import (
    compute_loss,
)


def build_test_config():

    return LlamaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def create_test_batch():

    return jnp.arange(
        2 * 16,
        dtype=jnp.int32,
    ).reshape(2, 16)


# ============================================================
# Forward compile
# ============================================================

def test_forward_compile():

    config = build_test_config()

    model = LlamaForCausalLM(
        config=config
    )

    rng = jax.random.PRNGKey(0)

    batch = create_test_batch()

    variables = model.init(
        rng,
        input_ids=batch,
        use_cache=False,
        mode="train",
    )

    logits, _ = model.apply(
        variables,
        input_ids=batch,
        use_cache=False,
        mode="train",
    )

    assert logits.shape == (
        2,
        16,
        config.vocab_size,
    )

    assert logits.dtype == (
        config.output_dtype
    )


# ============================================================
# Loss compile
# ============================================================

def test_loss_compile():

    config = build_test_config()

    logits = jnp.ones(
        (
            2,
            16,
            config.vocab_size,
        ),
        dtype=jnp.float32,
    )

    targets = jnp.ones(
        (
            2,
            16,
        ),
        dtype=jnp.int32,
    )

    loss, metrics = compute_loss(
        logits,
        targets,
    )

    assert loss.shape == ()

    assert "loss" in metrics

    assert "z_loss" in metrics


# ============================================================
# Backward compile
# ============================================================

def test_backward_compile():

    config = build_test_config()

    model = LlamaForCausalLM(
        config=config
    )

    rng = jax.random.PRNGKey(0)

    batch = create_test_batch()

    variables = model.init(
        rng,
        input_ids=batch,
        use_cache=False,
        mode="train",
    )

    params = variables["params"]

    def loss_fn(params):

        logits, _ = model.apply(
            {"params": params},
            input_ids=batch[:, :-1],
            use_cache=False,
            mode="train",
        )

        loss, _ = compute_loss(
            logits,
            batch[:, 1:],
        )

        return loss

    grads = jax.grad(
        loss_fn
    )(params)

    assert grads is not None