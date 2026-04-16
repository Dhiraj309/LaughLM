import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict

from jax.sharding import PartitionSpec as P, NamedSharding

from LaughLM.training.loss import shift_tokens, compute_loss


Params = Any
OptState = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ------------------------------------------------------------
# TRUE GRADIENT ACCUMULATION (SCAN-BASED)
# ------------------------------------------------------------

def create_train_step(model, optimizer, grad_accum: int, mesh=None) -> Callable:

    def loss_fn(params: Params, batch: Batch):
        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)
        loss, _ = compute_loss(logits, targets)
        return loss

    def micro_step(carry, micro_batch):
        params, grads_accum = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, micro_batch)
        grads_accum = jax.tree_util.tree_map(
            lambda g_acc, g: g_acc + g, grads_accum, grads
        )
        return (params, grads_accum), loss

    def train_step(params: Params, opt_state: OptState, batch: Batch):
        """
        batch shape: [grad_accum, B_global, T]
        B_global is sharded across the 'data' mesh axis.
        Each device sees [grad_accum, B_local, T].
        """
        grads_init = jax.tree_util.tree_map(jnp.zeros_like, params)

        (_, grads), losses = jax.lax.scan(micro_step, (params, grads_init), batch)

        grads = jax.tree_util.tree_map(lambda g: g / grad_accum, grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = {"loss": jnp.mean(losses)}
        
        # loss = jnp.mean(losses)
        
        # loss = loss = jax.lax.stop_gradient(
        #     jax.tree_util.tree_reduce(
        #         lambda a, b : a + b.reshape(-1)[0],
        #         new_params,
        #         initializer=0.0
        #         )
        # ) * 0.0
        
        # metrics = {"loss": loss}

        return new_params, new_opt_state, metrics

    # ------------------------------------------------------------------
    # JIT — explicit sharding contract when mesh is provided
    # ------------------------------------------------------------------
    if mesh is not None:
        replicated = NamedSharding(mesh, P())
        # batch: [grad_accum, B_global, T]
        # dim 0 (grad_accum) — not sharded
        # dim 1 (B_global) — sharded across 'data' axis
        # dim 2 (T) — not sharded
        batch_shard = NamedSharding(mesh, P(None, 'data', None))

        return jax.jit(
            train_step,
            donate_argnums=(0, 1),
            in_shardings=(replicated, replicated, batch_shard),
            out_shardings=(replicated, replicated, replicated),
        )

    # Fallback: no mesh (single-device / test runs)
    return jax.jit(train_step, donate_argnums=(0, 1))


# ------------------------------------------------------------
# EVAL STEP
# ------------------------------------------------------------

def create_eval_step(model):

    def eval_step(params, batch):
        inputs, targets = shift_tokens(batch)
        logits = model.apply({"params": params}, inputs)

        loss, metrics = compute_loss(logits, targets)

        # 🔥 FIX: include loss
        metrics = dict(metrics)
        metrics["loss"] = loss

        return metrics

    return jax.jit(eval_step)