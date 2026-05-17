"""
tests/test_tiny_overfit.py

Tiny overfit test for LaughLM Llama.

Purpose
-------
Verifies:
- gradients flow correctly
- causal masking works
- logits are stable
- residual paths are correct
- RoPE participates correctly in training
- KV-free training path is stable
- optimizer integration works

Expected behavior
-----------------
Loss should collapse rapidly.

Typical:
    step 0   -> ~4-5
    step 200 -> <0.1
"""

import jax
import jax.numpy as jnp
import optax

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)


def build_test_config():

    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """
    logits:
        [B, T, V]

    labels:
        [B, T]
    """

    log_probs = jax.nn.log_softmax(
        logits,
        axis=-1,
    )

    token_log_probs = jnp.take_along_axis(
        log_probs,
        labels[..., None],
        axis=-1,
    )

    token_log_probs = token_log_probs.squeeze(-1)

    return -jnp.mean(token_log_probs)


def test_tiny_overfit():

    config = build_test_config()

    model = LlamaForCausalLM(config)

    rng = jax.random.PRNGKey(0)

    # --------------------------------------------------
    # Tiny deterministic dataset
    # --------------------------------------------------

    input_ids = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=jnp.int32,
    )

    labels = jnp.array(
        [[2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=jnp.int32,
    )

    position_ids = jnp.arange(
        input_ids.shape[1],
        dtype=jnp.int32,
    )[None, :]

    # --------------------------------------------------
    # Initialize
    # --------------------------------------------------

    variables = model.init(
        rng,
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
        mode="train",
    )

    params = variables["params"]

    optimizer = optax.adamw(
        learning_rate=1e-3,
    )

    opt_state = optimizer.init(params)

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    def loss_fn(params):

        logits, _ = model.apply(
            {"params": params},
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            mode="train",
        )

        loss = cross_entropy_loss(
            logits,
            labels,
        )

        return loss

    # --------------------------------------------------
    # Train step
    # --------------------------------------------------

    @jax.jit
    def train_step(
        params,
        opt_state,
    ):

        loss, grads = jax.value_and_grad(
            loss_fn
        )(params)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        return params, opt_state, loss

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    initial_loss = None

    final_loss = None

    for step in range(300):

        params, opt_state, loss = train_step(
            params,
            opt_state,
        )

        loss = float(loss)

        if step == 0:
            initial_loss = loss

        final_loss = loss

        if step % 50 == 0:

            print(
                f"step={step:<3d} "
                f"loss={loss:.6f}"
            )

    print()

    print(
        f"initial_loss={initial_loss:.6f}"
    )

    print(
        f"final_loss={final_loss:.6f}"
    )


    from flax.traverse_util import flatten_dict

    print()
    print("=" * 80)
    print("PARAMETER TREE")
    print("=" * 80)
    
    flat = flatten_dict(variables["params"])
    
    for k, v in flat.items():
        print(k, v.shape)

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert final_loss < 0.05

    assert final_loss < initial_loss