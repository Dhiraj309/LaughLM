
import jax
import jax.numpybas jnp
import optax

from LaughLM.training.loss import shift_tokens, compute_loss

def create_train_step(model, optimizer):
    """
    Build the compiled training step.

    Returns a function:
        train_step(params, opt_state, batch)
    """

    def loss_fn(params, batch):
        inputs, targets = shift_tokens(batch)

        logits = model.apply(
            {"params": params},
            inputs
        )

        loss, metrics = compute_loss(logits, targets)

        return loss, (metrics, logits)

    def train_state(param, opt_state, batch):
        (loss, (metrics, logits)), grads = jax.grad_and_value(
            loss_fn,
            has_aux=True
        )(params, batch)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        metrics["loss"] = loss

        return params, out_state, metrics

    return jax.jit(train_step)
