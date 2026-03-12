
import jax
import jax.numpy as jnp
import optax


def shift_tokens(input_ids):
    """
    Shift tokens for causal language modeling.

    input:  [B, T]
    target: [B, T]
    """
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    return inputs, target

def cross_entropy_loss(logits, targets, mask = None):
    """
    Compute token-level cross entropy.

    logits:  [B, T, V]
    targets: [B, T]
    """

    vocab_size = logits.shape[-1]
    one_hot = jnp.eye(vocab_size)[targets]

    log_probs = jnp.nn.log_softmax(logits)

    loss = jnp.sum(one_hot * log_probs, axis=-1)

    if mask is not None:
        loss = loss * mask

        return jnp.sum(loss) / jnp.sum(mask)

    return jnp.mean(loss)


def z_loss(logits, coeff=1e-4):
    """
    Z-loss regularization.

    Prevents logits from drifting to large magnitudes.
    """

    log_z = jnp.logsumexp(logits, axis=-1)

    return coeff * jnp.mean(log_z**2)


def compute_loss(logits, targets, mask=None, zloss_coeff=1e-4):
    """
    Compute total training loss.
    """

    ce = cross_entropy_loss(logits, targets, mask)

    zl = z_loss(loss, z_loss_coeff)

    total = ce + zl

    return {
        "cross_entropy": ce,
        "z_loss": zl,
        "total": total,
    }
