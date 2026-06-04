import jax
import jax.numpy as jnp

from LaughLM.training.train_state import TrainState


def test_train_state_apply_grad_step_increments_step_without_tokens():
    state = TrainState(
        params={"w": jnp.asarray(1.0)},
        opt_state={"m": jnp.asarray(0.0)},
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int32),
        rng_key=None,
    )

    new_state = state.apply_grad_step(
        params={"w": jnp.asarray(2.0)},
        opt_state={"m": jnp.asarray(1.0)},
    )

    assert int(jax.device_get(new_state.step)) == 1
    assert int(jax.device_get(new_state.tokens_processed)) == 0


def test_train_state_apply_grad_step_increments_step_and_tokens():
    state = TrainState(
        params={"w": jnp.asarray(1.0)},
        opt_state={"m": jnp.asarray(0.0)},
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int32),
        rng_key=None,
    )

    new_state = state.apply_grad_step(
        params={"w": jnp.asarray(2.0)},
        opt_state={"m": jnp.asarray(1.0)},
        tokens_in_step=jnp.asarray(128, dtype=jnp.int32),
    )

    assert int(jax.device_get(new_state.step)) == 1
    assert int(jax.device_get(new_state.tokens_processed)) == 128


def test_train_state_apply_grad_step_accumulates_existing_tokens():
    state = TrainState(
        params={"w": jnp.asarray(1.0)},
        opt_state={"m": jnp.asarray(0.0)},
        step=jnp.asarray(7, dtype=jnp.int32),
        tokens_processed=jnp.asarray(1024, dtype=jnp.int32),
        rng_key=None,
    )

    new_state = state.apply_grad_step(
        params={"w": jnp.asarray(2.0)},
        opt_state={"m": jnp.asarray(1.0)},
        tokens_in_step=jnp.asarray(256, dtype=jnp.int32),
    )

    assert int(jax.device_get(new_state.step)) == 8
    assert int(jax.device_get(new_state.tokens_processed)) == 1280


def test_train_state_apply_grad_step_replaces_extra_state_only_when_provided():
    state = TrainState(
        params={"w": jnp.asarray(1.0)},
        opt_state={"m": jnp.asarray(0.0)},
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int32),
        rng_key=None,
        extra_state={"old": True},
    )

    unchanged_extra = state.apply_grad_step(
        params={"w": jnp.asarray(2.0)},
        opt_state={"m": jnp.asarray(1.0)},
    )

    replaced_extra = state.apply_grad_step(
        params={"w": jnp.asarray(3.0)},
        opt_state={"m": jnp.asarray(2.0)},
        extra_state={"new": True},
    )

    assert unchanged_extra.extra_state == {"old": True}
    assert replaced_extra.extra_state == {"new": True}


def test_train_state_next_rng_advances_key_and_returns_subkey():
    key = jax.random.PRNGKey(0)

    state = TrainState(
        params={"w": jnp.asarray(1.0)},
        opt_state={"m": jnp.asarray(0.0)},
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int32),
        rng_key=key,
    )

    new_state, subkey = state.next_rng()

    assert new_state.rng_key.shape == key.shape
    assert subkey.shape == key.shape

    assert not bool(
        jnp.all(
            jax.device_get(
                new_state.rng_key == state.rng_key
            )
        )
    )
