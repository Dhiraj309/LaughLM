import time
from typing import Iterator

import jax
import jax.numpy as jnp
import optax

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.parameter_utils import generate_preflight_report
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler, compute_total_steps
from LaughLM.training.train_step import create_train_step, create_eval_step
from LaughLM.training.logger import TrainingLogger
from LaughLM.utils.rng import create_rng


class Trainer:
    """
    LaughLM training orchestration.

    Handles:
      - Model + optimizer initialization
      - JIT-compiled training and eval steps
      - Gradient accumulation across micro-batches
      - Logging (loss, LR, tokens/sec, per-component metrics)
      - Checkpoint saving / resuming

    Fixes applied
    -------------
    FIX 1: self.config = LaughLMConfig  →  self.config = config
            (was assigning the class, not the instance)

    FIX 2: self.optimizer(params)  →  self.optimizer.init(params)
            (Optax optimizers use .init(), not direct call)

    FIX 3: config.  →  self.config.  in train() body
            ('config' was not in scope — 'self.config' is)

    FIX 4: Gradient accumulation implemented
            (was in config but completely absent from training loop)

    FIX 5: Schedule is built first, then passed to build_optimizer
            (previously schedule was separate object, LR was never applied)
    """

    def __init__(self, config: LaughLMConfig):

        # FIX 1: store instance, not class
        self.config = config

        self.rng = create_rng(seed=42)

        # Print parameter / memory / step estimates before training starts
        generate_preflight_report(config)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        self.model = GPTModel(config=config)

        # Initialize parameters with a dummy batch
        dummy = jnp.zeros(
            (config.runtime.micro_batch_per_device, config.runtime.seq_len),
            dtype=jnp.int32,
        )
        self.params = self.model.init(self.rng.next_key(), dummy)["params"]

        # ------------------------------------------------------------------
        # Schedule (built first, passed to optimizer)
        # FIX 5: schedule is baked into optimizer chain via
        #        optax.scale_by_learning_rate(schedule)
        # ------------------------------------------------------------------
        self.schedule  = build_scheduler(config)
        self.optimizer = build_optimizer(config, self.schedule)

        # FIX 2: initialize optimizer state with .init()
        self.opt_state = self.optimizer.init(self.params)

        # ------------------------------------------------------------------
        # Compiled step functions
        # ------------------------------------------------------------------
        self.train_step = create_train_step(self.model, self.optimizer)
        self.eval_step  = create_eval_step(self.model)

        # ------------------------------------------------------------------
        # Step counter
        # ------------------------------------------------------------------
        self.step = 0

        # ------------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------------
        from LaughLM.model.parameter_utils import estimate_parameters
        total_params = estimate_parameters(config)["total_params"]
        self.logger = TrainingLogger(config, total_params=total_params)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader: Iterator) -> None:
        """
        Main training loop with gradient accumulation.

        Gradient accumulation
        ---------------------
        We cannot fit large batches in a single TPU step, so we split
        each logical batch into N micro-batches, accumulate their gradients,
        then apply one optimizer update.

        Effective batch size = micro_batch × data_parallel × grad_accum × seq_len
        Example (this config): 32 × 8 × 16 × 2048 = 8,388,608 tokens/step

        This is critical. Without accumulation we get ~32K tokens/step, which is
        far below the ~4M token/step that Chinchilla-optimal training requires
        for stable, efficient learning at 143M scale.
        """

        cfg = self.config  # FIX 3: use self.config, not bare 'config'

        total_steps     = compute_total_steps(cfg)
        grad_accum      = cfg.runtime.gradient_accumulation
        tokens_per_step = (
            cfg.runtime.seq_len
            * cfg.runtime.micro_batch_per_device
            * cfg.parallelism.data_parallel
            * grad_accum
        )

        print(f"\n{'='*60}")
        print(f"  Training for {total_steps:,} optimizer steps")
        print(f"  Gradient accumulation: {grad_accum} micro-batches")
        print(f"  Effective tokens per step: {tokens_per_step:,}")
        print(f"{'='*60}\n")

        start_time  = time.time()
        accum_grads = None   # Accumulated gradient tree
        accum_loss  = 0.0    # Sum of micro-batch losses for logging

        for micro_step, batch in enumerate(dataloader):

            # ----------------------------------------------------------------
            # Compute gradients for this micro-batch (no optimizer update yet)
            # ----------------------------------------------------------------
            loss, metrics, grads, grad_norm = self._compute_grads(batch)
            accum_loss += loss

            # Accumulate gradients: sum across micro-batches
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = jax.tree_util.tree_map(
                    lambda a, b: a + b,
                    accum_grads,
                    grads,
                )

            # ----------------------------------------------------------------
            # Apply update every `grad_accum` micro-batches
            # ----------------------------------------------------------------
            is_update_step = ((micro_step + 1) % grad_accum == 0)

            if is_update_step:

                # Average gradients over accumulation steps
                accum_grads = jax.tree_util.tree_map(
                    lambda g: g / grad_accum,
                    accum_grads,
                )

                # Optimizer update (clip → adam → weight_decay → lr_scale)
                updates, self.opt_state = self.optimizer.update(
                    accum_grads,
                    self.opt_state,
                    self.params,
                )
                self.params = optax.apply_updates(self.params, updates)

                # Reset accumulation state
                accum_grads = None
                avg_loss    = accum_loss / grad_accum
                accum_loss  = 0.0

                self.step += 1

                # ----------------------------------------------------------------
                # Step logging (phase banners fire automatically inside log_step)
                # ----------------------------------------------------------------
                if self.step % cfg.runtime.log_interval == 0:
                    lr          = float(self.schedule(self.step))
                    tokens_seen = self.step * tokens_per_step
                    metrics["loss"] = avg_loss
                    # grad_norm already computed in _compute_grads

                    self.logger.log_step(
                        step=self.step,
                        metrics=metrics,
                        lr=lr,
                        grad_norm=grad_norm,
                        tokens_seen=tokens_seen,
                    )

                # ----------------------------------------------------------------
                # Eval
                # ----------------------------------------------------------------
                if self.step % cfg.runtime.eval_interval == 0:
                    self._run_eval()

                # ----------------------------------------------------------------
                # Done?
                # ----------------------------------------------------------------
                if self.step >= total_steps:
                    break

        self.logger.log_summary(self.step, self.step * tokens_per_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_grads(self, batch):
        """
        Compute loss, gradients, and grad norm for a single micro-batch.
        Does NOT apply the optimizer update.

        Grad norm is computed BEFORE the optimizer clips it, which is
        what you want — the pre-clip norm is the real instability signal.
        """
        from LaughLM.training.loss import shift_tokens, compute_loss

        def loss_fn(params):
            inputs, targets = shift_tokens(batch)
            logits = self.model.apply({"params": params}, inputs)
            loss, metrics = compute_loss(logits, targets)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.params)

        # Global L2 grad norm across all parameters (pre-clip)
        grad_norm = float(
            jnp.sqrt(
                sum(
                    jnp.sum(g ** 2)
                    for g in jax.tree_util.tree_leaves(grads)
                )
            )
        )

        # Only convert loss/metrics to float at log time to minimise
        # device syncs. grad_norm sync is unavoidable here.
        return float(loss), metrics, grads, grad_norm

    def _run_eval(self):
        """
        Run a quick eval pass on a fixed held-out batch.
        In a real training loop, pass a separate eval dataloader.
        Placeholder here — replace with real eval data.
        """
        # TODO: wire up real eval dataloader
        print(f"  [Eval] step={self.step} — plug in eval dataloader")

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save params + optimizer state + step counter.

        For WSD: save at end of STABLE phase (before decay).
        This checkpoint can be resumed for extended training
        by loading params + opt_state and continuing with a new schedule.
        """
        import pickle
        ckpt = {
            "params":    self.params,
            "opt_state": self.opt_state,
            "step":      self.step,
        }
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"Checkpoint saved → {path} (step {self.step:,})")

    def load_checkpoint(self, path: str) -> None:
        """
        Load params + optimizer state + step counter.

        When resuming WSD training from the STABLE phase checkpoint:
          - Load this checkpoint
          - Build a new schedule with warmup_steps=0 and extended total_tokens
          - The resumed run enters stable phase immediately (no re-warmup)
        """
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.params    = ckpt["params"]
        self.opt_state = ckpt["opt_state"]
        self.step      = ckpt["step"]
        print(f"Checkpoint loaded ← {path} (step {self.step:,})")
