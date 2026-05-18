# tests/test_tiny_overfit.py

import jax
import jax.numpy as jnp
import optax

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
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )


def test_tiny_overfit():

    config = build_test_config()

    model = LlamaForCausalLM(
        config=config
    )

    rng = jax.random.PRNGKey(0)

    tokens = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=jnp.int32,
    )

    variables = model.init(
        rng,
        input_ids=tokens[:, :-1],
        use_cache=False,
        mode="train",
    )

    params = variables["params"]

    optimizer = optax.adamw(
        learning_rate=1e-3,
    )

    opt_state = optimizer.init(
        params
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    def loss_fn(params):

        logits, _ = model.apply(
            {"params": params},
            input_ids=tokens[:, :-1],
            use_cache=False,
            mode="train",
        )

        loss, _ = compute_loss(
            logits,
            tokens[:, 1:],
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

        return (
            params,
            opt_state,
            loss,
        )

    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------

    initial_loss = None
    final_loss = None

    for step in range(300):

        (
            params,
            opt_state,
            loss,
        ) = train_step(
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

    assert final_loss < initial_loss

    assert final_loss < 0.1