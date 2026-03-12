import time
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.parameter_utils import generate_preflight_report

from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler
from LaughLM.training.train_step import create_train_step, create_grad_fn

from LaughLM.utils.rng import create_rng


class Trainer:
    """
    LaughLM training orchestration.
    """

    def __init__(self, config: LaughLMConfig):

        # ── FIX 1: was `self.config = LaughLMConfig` (assigned the class, not the instance) ──
        self.config = config

        self.rng = create_rng(seed=42)

        generate_preflight_report(config)

        # ── Model ────────────────────────────────────────────────────────────
        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (
                config.runtime.micro_batch_per_device,
                config.runtime.seq_len,
            ),
            dtype=jnp.int32,
        )

        params = self.model.init(self.rng.next_key(), dummy)["params"]

        # ── FIX 2: mixed precision — store params as float32 (master weights)
        #    The forward pass will cast to bfloat16 internally.
        #    This is the standard AMP setup: float32 params + bf16 compute.
        self.params = params

        # ── Optimizer + Scheduler ────────────────────────────────────────────
        self.schedule = build_scheduler(config)

        # build_optimizer now chains schedule internally (fixed in optimizer.py)
        self.optimizer = build_optimizer(config)

        # ── FIX 3: was `self.optimizer(params)` — Optax uses .init(), not __call__ ──
        self.opt_state = self.optimizer.init(params)

        # ── Compiled step functions ──────────────────────────────────────────
        # Separate grad function from update function to support accumulation
        self.grad_fn   = create_grad_fn(self.model)
        self.train_step = create_train_step(self.model, self.optimizer)

        # ── Step counter ─────────────────────────────────────────────────────
        self.global_step = 0          # optimizer update steps (after accumulation)
        self.micro_step  = 0          # total forward passes
        self.tokens_seen = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Token / step bookkeeping
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_total_steps(self) -> int:
        cfg = self.config
        tokens_per_micro = (
            cfg.runtime.seq_len
            * cfg.runtime.micro_batch_per_device
            * cfg.parallelism.data_parallel
        )
        total_micro_steps = cfg.runtime.total_tokens // tokens_per_micro
        return total_micro_steps // cfg.runtime.gradient_accumulation

    def _tokens_per_optimizer_step(self) -> int:
        cfg = self.config
        return (
            cfg.runtime.seq_len
            * cfg.runtime.micro_batch_per_device
            * cfg.parallelism.data_parallel
            * cfg.runtime.gradient_accumulation
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient accumulation helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_like(tree):
        """Create a zeroed pytree with the same structure as `tree`."""
        return jtu.tree_map(jnp.zeros_like, tree)

    @staticmethod
    def _accumulate(accum, new):
        """Element-wise add two pytrees."""
        return jtu.tree_map(lambda a, b: a + b, accum, new)

    @staticmethod
    def _scale(tree, factor: float):
        """Scale every leaf in a pytree by `factor`."""
        return jtu.tree_map(lambda x: x * factor, tree)

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, dataloader):
        """
        Full training loop with gradient accumulation and mixed precision.

        Gradient accumulation
        ─────────────────────
        Every `gradient_accumulation` micro-batches are processed before a
        single optimizer update. This lets you reach millions of tokens per
        optimizer step without OOM.

        On v5e-8 with seq_len=2048, micro_batch=32, grad_accum=16:
            tokens/step = 2048 × 32 × 8 × 16 = 8,388,608 ≈ 8M tokens/step
        """

        cfg              = self.config
        accum_steps      = cfg.runtime.gradient_accumulation
        total_opt_steps  = self._compute_total_steps()
        tokens_per_step  = self._tokens_per_optimizer_step()

        print(f"\n{'─'*56}")
        print(f"  Training Configuration")
        print(f"{'─'*56}")
        print(f"  Optimizer steps    : {total_opt_steps:,}")
        print(f"  Gradient accum     : {accum_steps}")
        print(f"  Tokens/opt step    : {tokens_per_step:,}")
        print(f"  Total tokens       : {cfg.runtime.total_tokens:,}")
        print(f"{'─'*56}\n")

        # Accumulation state
        accum_grads   = self._zero_like(self.params)
        accum_metrics = {}

        start      = time.time()
        step_start = time.time()

        for micro_step, batch_tokens in enumerate(dataloader):

            # ── FIX 4: mixed precision — cast batch to int32, model handles bf16 ──
            batch_tokens = jnp.array(batch_tokens, dtype=jnp.int32)

            # ── Compute gradients (no optimizer update yet) ───────────────────
            grads, metrics = self.grad_fn(self.params, batch_tokens)

            # ── Accumulate gradients ──────────────────────────────────────────
            accum_grads = self._accumulate(accum_grads, grads)

            # Accumulate metrics for averaging
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v)

            self.micro_step  += 1
            self.tokens_seen += (
                cfg.runtime.seq_len
                * cfg.runtime.micro_batch_per_device
                * cfg.parallelism.data_parallel
            )

            # ── Optimizer update every `accum_steps` micro-batches ────────────
            if (micro_step + 1) % accum_steps == 0:

                # Average accumulated gradients
                averaged_grads = self._scale(accum_grads, 1.0 / accum_steps)

                # Apply optimizer update
                self.params, self.opt_state = self._apply_update(
                    self.params, self.opt_state, averaged_grads
                )

                # Average accumulated metrics
                avg_metrics = {k: v / accum_steps for k, v in accum_metrics.items()}

                self.global_step += 1

                # ── FIX 5: was `self.step % self config.` (missing dot) ───────
                #    was `config.runtime.log_interval` (bare name, not self.config)
                if self.global_step % self.config.runtime.log_interval == 0:
                    self._log(avg_metrics, tokens_per_step, step_start)
                    step_start = time.time()

                # Eval / checkpoint interval
                if self.global_step % self.config.runtime.eval_interval == 0:
                    self._checkpoint(avg_metrics)

                # Reset accumulation buffers
                accum_grads   = self._zero_like(self.params)
                accum_metrics = {}

                # ── Stop condition ────────────────────────────────────────────
                if self.global_step >= total_opt_steps:
                    break

        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed/3600:.2f}h — "
              f"{self.tokens_seen:,} tokens over {self.global_step:,} steps")

    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer apply (separated for gradient accumulation)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_update(self, params, opt_state, grads):
        """Apply optimizer update. Returns (new_params, new_opt_state)."""
        import optax
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, metrics: dict, tokens_per_step: int, step_start: float):
        """
        Print training metrics.

        Logs:
          step        — global optimizer step
          loss        — total training loss
          ce / kl     — CE and distillation components if present
          lr          — current learning rate from schedule
          tok/s       — tokens per second (wall-clock)
          tokens      — cumulative tokens seen
          grad_norm   — gradient norm (stability canary)
        """
        elapsed_step = max(time.time() - step_start, 1e-6)

        # ── FIX 6: tok/s was computing wrong value (used step count not time) ──
        tok_per_sec = tokens_per_step / elapsed_step

        # Current LR from schedule (schedule is a function of global step)
        lr = float(self.schedule(self.global_step))

        parts = [
            f"step={self.global_step:>6}",
            f"loss={metrics.get('total', metrics.get('loss', 0.0)):.4f}",
        ]

        if "cross_entropy" in metrics:
            parts.append(f"ce={metrics['cross_entropy']:.4f}")
        if "kl_loss" in metrics:
            parts.append(f"kl={metrics['kl_loss']:.4f}")
        if "grad_norm" in metrics:
            parts.append(f"‖g‖={metrics['grad_norm']:.3f}")

        parts += [
            f"lr={lr:.2e}",
            f"tok/s={tok_per_sec:,.0f}",
            f"tokens={self.tokens_seen/1e9:.3f}B",
        ]

        print("  ".join(parts))

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def _checkpoint(self, metrics: dict):
        """
        Save checkpoint. Placeholder — wire to orbax or simple numpy save.
        """
        print(
            f"  [ckpt] step={self.global_step} "
            f"loss={metrics.get('total', 0.0):.4f} "
            f"tokens={self.tokens_seen/1e9:.2f}B"
        )
        # TODO: implement orbax checkpoint save
        # import orbax.checkpoint as ocp
        # checkpointer = ocp.StandardCheckpointer()
        # checkpointer.save(f"checkpoints/step_{self.global_step}", self.params)
