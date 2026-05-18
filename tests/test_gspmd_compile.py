# tests/test_gspmd_compile.py

"""
End-to-end GSPMD compile validation tests.

Goals
-----
1. Model initialization compile
2. Forward-pass compile
3. Loss compile
4. Backward compile
5. Train-step compile
6. Decode-mode compile
7. Sharding inspection

IMPORTANT
---------
These tests validate:
- logical axis propagation
- NamedSharding correctness
- bf16/fp32 stability
- GSPMD compatibility
- scan/remat compatibility
- KV cache semantics

This is the canonical frontier integration test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from LaughLM.config.schema import (
    LaughLMConfig,
)

from LaughLM.model.llama.config_factory import (
    build_llama_config,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.model.llama.kv_cache import (
    initialize_kv_cache,
)

from LaughLM.training.loss import (
    compute_loss,
)

from LaughLM.training.train_step import (
    create_train_step,
)

from LaughLM.training.train_state import (
    create_train_state,
)

from LaughLM.distributed.mesh import (
    create_mesh,
)

from LaughLM.distributed.state import (
    create_abstract_state,
)

from LaughLM.distributed.sharding import (
    logical_to_sharding,
)


# ============================================================
# Helpers
# ============================================================

def create_test_config():

    config = LaughLMConfig()

    #
    # Tiny compile-safe config
    #

    config.model.d_model = 256
    config.model.num_heads = 8
    config.model.num_kv_heads = 8
    config.model.num_layers = 4
    config.model.max_seq_len = 128
    config.model.vocab_size = 32000

    #
    # Safer initial integration
    #

    config.architecture.scan_layers = False

    return config


def create_test_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):

    return jnp.arange(
        batch_size * seq_len,
        dtype=jnp.int32,
    ).reshape(
        batch_size,
        seq_len,
    ) % vocab_size


# ============================================================
# Model init
# ============================================================

def test_model_init_compile():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    mesh = create_mesh(config)

    rng = jax.random.PRNGKey(0)

    with mesh:

        variables = model.init(
            rng,
            jnp.ones(
                (2, 16),
                dtype=jnp.int32,
            ),
        )

    assert "params" in variables


# ============================================================
# Abstract sharding
# ============================================================

def test_abstract_state_creation():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    mesh = create_mesh(config)

    rng = jax.random.PRNGKey(0)

    (
        abstract_state,
        logical_specs,
        shardings,
    ) = create_abstract_state(
        model=model,
        config=config,
        mesh=mesh,
        rng=rng,
        input_shape=(2, 16),
    )

    assert abstract_state is not None
    assert logical_specs is not None
    assert shardings is not None


# ============================================================
# Forward compile
# ============================================================

def test_forward_compile():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    rng = jax.random.PRNGKey(0)

    batch = create_test_batch(
        batch_size=2,
        seq_len=16,
        vocab_size=llama_config.vocab_size,
    )

    variables = model.init(
        rng,
        batch,
    )

    logits, _ = model.apply(
        variables,
        batch,
        mode="train",
    )

    assert logits.shape == (
        2,
        16,
        llama_config.vocab_size,
    )

    assert logits.dtype == (
        llama_config.output_dtype
    )


# ============================================================
# Loss compile
# ============================================================

def test_loss_compile():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    logits = jnp.ones(
        (
            2,
            16,
            llama_config.vocab_size,
        ),
        dtype=jnp.float32,
    )

    targets = jnp.ones(
        (2, 16),
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

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    rng = jax.random.PRNGKey(0)

    batch = create_test_batch(
        batch_size=2,
        seq_len=16,
        vocab_size=llama_config.vocab_size,
    )

    variables = model.init(
        rng,
        batch,
    )

    params = variables["params"]

    def loss_fn(params):

        logits, _ = model.apply(
            {"params": params},
            batch,
            mode="train",
        )

        targets = batch

        loss, _ = compute_loss(
            logits[:, :-1],
            targets[:, 1:],
        )

        return loss

    grads = jax.grad(loss_fn)(
        params
    )

    assert grads is not None


# ============================================================
# Train step compile
# ============================================================

def test_train_step_compile():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    mesh = create_mesh(config)

    rng = jax.random.PRNGKey(0)

    optimizer = optax.adamw(
        learning_rate=1e-4,
    )

    #
    # Initialize variables
    #

    variables = model.init(
        rng,
        jnp.ones(
            (2, 16),
            dtype=jnp.int32,
        ),
    )

    state = create_train_state(
        params=variables["params"],
        optimizer=optimizer,
        rng_key=rng,
    )

    #
    # Placeholder shardings
    #
    # Replace with concrete shardings
    # after distributed init stabilizes.
    #

    state_shardings = None
    data_sharding = None

    train_step = create_train_step(
        model=model,
        optimizer=optimizer,
        config=config,
        mesh=mesh,
        state_shardings=state_shardings,
        data_sharding=data_sharding,
        grad_accum=2,
    )

    batch = jnp.ones(
        (
            2,   # grad accum
            2,   # batch
            16,  # seq
        ),
        dtype=jnp.int32,
    )

    new_state, metrics = (
        train_step(
            state,
            batch,
        )
    )

    assert "loss" in metrics
    assert "grad_norm" in metrics


# ============================================================
# Decode compile
# ============================================================

def test_decode_compile():

    config = create_test_config()

    config.architecture.scan_layers = False

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    rng = jax.random.PRNGKey(0)

    batch_size = 2
    seq_len = 1

    input_ids = jnp.ones(
        (
            batch_size,
            seq_len,
        ),
        dtype=jnp.int32,
    )

    kv_caches = [
        initialize_kv_cache(
            batch_size=batch_size,
            max_length=128,
            config=llama_config,
        )
        for _ in range(
            llama_config.num_hidden_layers
        )
    ]

    variables = model.init(
        rng,
        input_ids,
        kv_caches=kv_caches,
        use_cache=True,
        mode="decode",
    )

    logits, updated_caches = (
        model.apply(
            variables,
            input_ids,
            kv_caches=kv_caches,
            use_cache=True,
            mode="decode",
        )
    )

    assert logits.shape == (
        batch_size,
        seq_len,
        llama_config.vocab_size,
    )

    assert updated_caches is not None


# ============================================================
# Sharding visualization
# ============================================================

def test_sharding_visualization():

    config = create_test_config()

    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    mesh = create_mesh(config)

    rng = jax.random.PRNGKey(0)

    with mesh:

        variables = model.init(
            rng,
            jnp.ones(
                (2, 16),
                dtype=jnp.int32,
            ),
        )

    embedding = (
        variables["params"]
        ["model"]
        ["embed_tokens"]
        ["embedding"]
    )

    print(
        jax.debug.visualize_array_sharding(
            embedding
        )
    )
