# tests/test_tpu_train_step.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax.sharding import (
    Mesh,
    NamedSharding,
    PartitionSpec as P,
)

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.training.fsdp_train_step import (
    create_fsdp_train_step,
)

from LaughLM.training.train_state import (
    TrainState,
)


# ============================================================
# Config
# ============================================================

def build_test_config():
    return LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


# ============================================================
# Batch
# ============================================================

def create_batch():
    """
    Shape:
        [grad_accum, batch, seq]
    """

    return (
        jnp.arange(
            2 * 2 * 16,
            dtype=jnp.int32,
        )
        .reshape(
            2,
            2,
            16,
        )
        % 256
    )


# ============================================================
# Mesh train step
# ============================================================

def test_tpu_train_step():
    config = build_test_config()

    model = LlamaForCausalLM(
        config=config,
    )

    rng = jax.random.PRNGKey(0)

    optimizer = optax.adamw(
        learning_rate=1e-3,
    )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------
    #
    # Use a single-device mesh for this unit test so the test is portable
    # across CPU/GPU/TPU and does not require batch-size divisibility by
    # all available devices.
    #
    # This test validates the mesh-jit train-step API, not multi-device
    # throughput.
    # --------------------------------------------------------

    devices = np.asarray(
        jax.devices()[:1],
    )

    mesh = Mesh(
        devices,
        axis_names=("batch",),
    )

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    dummy_input = jnp.ones(
        (
            2,
            16,
        ),
        dtype=jnp.int32,
    )

    with mesh:
        variables = model.init(
            rng,
            input_ids=dummy_input,
            use_cache=False,
            mode="train",
        )

    params = variables["params"]

    opt_state = optimizer.init(
        params,
    )

    state = TrainState(
        params=params,
        opt_state=opt_state,
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int32),
        rng_key=rng,
    )

    # --------------------------------------------------------
    # Shardings
    # --------------------------------------------------------

    state_sharding = NamedSharding(
        mesh,
        P(),
    )

    batch_sharding = NamedSharding(
        mesh,
        P(
            None,
            "batch",
            None,
        ),
    )

    metrics_sharding = NamedSharding(
        mesh,
        P(),
    )

    # --------------------------------------------------------
    # Train step
    # --------------------------------------------------------

    train_step = create_fsdp_train_step(
        model=model,
        optimizer=optimizer,
        state_sharding=state_sharding,
        batch_sharding=batch_sharding,
        metrics_sharding=metrics_sharding,
        grad_accum=2,
        max_grad_norm=1.0,
    )

    batch = create_batch()

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    with mesh:
        new_state, metrics = train_step(
            state,
            batch,
        )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    expected_tokens = (
        batch.shape[0]
        * batch.shape[1]
        * batch.shape[2]
    )

    assert int(jax.device_get(new_state.step)) == 1

    assert (
        int(jax.device_get(new_state.tokens_processed))
        == expected_tokens
    )

    assert "loss" in metrics
    assert "grad_norm" in metrics

    loss = float(
        jax.device_get(metrics["loss"])
    )

    grad_norm = float(
        jax.device_get(metrics["grad_norm"])
    )

    print()
    print(
        f"loss={loss:.6f}"
    )
    print(
        f"grad_norm={grad_norm:.6f}"
    )

    assert np.isfinite(loss)
    assert np.isfinite(grad_norm)
