import json
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.parameter_utils import generate_preflight_report, estimate_parameters
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler, compute_total_steps
from LaughLM.training.train_step import create_train_step, create_eval_step
from LaughLM.training.logger import TrainingLogger
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.utils.rng import create_rng


# ------------------------------------------------------------
# Safe scalar conversion
# ------------------------------------------------------------

def _scalar(x):
    """
    Convert JAX scalars to Python floats safely.
    Prevents DeviceArray printing explosions.
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(jax.device_get(x))
        except Exception:
            return float("nan")


class Trainer:
    """
    LaughLM training orchestration with Orbax checkpointing.
    """

    def __init__(self, config: LaughLMConfig, resume_dir: str | None = None):

        self.config = config

        # ------------------------------------------------------------
        # RNG
        # ------------------------------------------------------------
        self.rng = create_rng(seed=42)

        # ------------------------------------------------------------
        # Preflight report
        # ------------------------------------------------------------
        generate_preflight_report(config)

        # ------------------------------------------------------------
        # Model
        # ------------------------------------------------------------
        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (
                config.runtime.micro_batch_per_device,
                config.runtime.seq_len,
            ),
            dtype=jnp.int32,
        )

        self.params = self.model.init(
            self.rng.next_key(),
            dummy,
        )["params"]

        # ------------------------------------------------------------
        # Optimizer + scheduler
        # ------------------------------------------------------------
        self.schedule = build_scheduler(config)
        self.optimizer = build_optimizer(config, self.schedule)

        self.opt_state = self.optimizer.init(self.params)

        # ------------------------------------------------------------
        # JIT steps
        # ------------------------------------------------------------
        self.train_step = create_train_step(self.model, self.optimizer)
        self.eval_step = create_eval_step(self.model)

        # ------------------------------------------------------------
        # Counters
        # ------------------------------------------------------------
        self.step = 0
        self.tokens_processed = 0

        # ------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------
        total_params = estimate_parameters(config)["total_params"]

        self.logger = TrainingLogger(
            config,
            total_params=total_params,
        )

        # ------------------------------------------------------------
        # Checkpoint manager
        # ------------------------------------------------------------
        ckpt_dir = resume_dir or config.runtime.checkpoint_dir

        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=config.runtime.checkpoint_max_to_keep,
        )

        self.checkpoint_interval = config.runtime.checkpoint_interval

        # ------------------------------------------------------------
        # Save config once
        # ------------------------------------------------------------
        config_path = Path(ckpt_dir) / "config.json"

        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

        # ------------------------------------------------------------
        # Restore checkpoint if available
        # ------------------------------------------------------------
        restored = self.checkpoints.restore_latest()

        if restored is not None:

            state, step = restored

            self.params = state["params"]

            """
            init_opt_state = self.optimizer.init(self.params)

            self.opt_state = jax.tree_util.tree_map(
                lambda new, old: old,
                init_opt_state,
                state["opt_state"],
            )"""

            self.opt_state = state["opt_state"]

            self.step = state["step"]
            self.tokens_processed = state["tokens_processed"]

            self.rng._key = state["rng_key"]

            print(f"[trainer] resumed from step {self.step}")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    def train(self, dataloader: Iterator):

        cfg = self.config

        total_steps = compute_total_steps(cfg)

        tokens_per_step = (
            cfg.runtime.seq_len
            * cfg.runtime.micro_batch_per_device
            * cfg.parallelism.data_parallel
        )

        print("\n" + "=" * 60)
        print(f"Training for {total_steps:,} optimizer steps")
        print(f"Effective tokens per step: {tokens_per_step:,}")
        print("=" * 60 + "\n")

        for batch in dataloader:

            batch = jnp.asarray(batch, dtype=jnp.int32)

            # --------------------------------------------------------
            # JIT compiled update
            # --------------------------------------------------------
            self.params, self.opt_state, metrics = self.train_step(
                self.params,
                self.opt_state,
                batch,
            )

            # Ensure metrics come back to CPU safely
            metrics = jax.device_get(metrics)

            self.step += 1
            self.tokens_processed += tokens_per_step

            # --------------------------------------------------------
            # Logging
            # --------------------------------------------------------
            if self.step % cfg.runtime.log_interval == 0:

                lr = _scalar(self.schedule(self.step))

                safe_metrics = {
                    k: _scalar(v)
                    for k, v in metrics.items()
                }

                self.logger.log_step(
                    step=self.step,
                    metrics=safe_metrics,
                    lr=lr,
                    grad_norm=None,
                    tokens_seen=self.tokens_processed,
                )

            # --------------------------------------------------------
            # Evaluation
            # --------------------------------------------------------
            if self.step % cfg.runtime.eval_interval == 0:
                self._run_eval()

            # --------------------------------------------------------
            # Checkpoint
            # --------------------------------------------------------
            if self.step % self.checkpoint_interval == 0:

                state = {
                    "params": self.params,
                    "opt_state": self.opt_state,
                    "step": self.step,
                    "tokens_processed": self.tokens_processed,
                    "rng_key": self.rng.key,
                }

                print(f"[checkpoint] saving step {self.step}")

                self.checkpoints.save(self.step, state)

            if self.step >= total_steps:
                break

        # ------------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------------
        final_state = {
            "params": self.params,
            "opt_state": self.opt_state,
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "rng_key": self.rng.key,
        }

        self.checkpoints.save(self.step, final_state)

        self.logger.log_summary(
            self.step,
            self.tokens_processed,
        )

    # ------------------------------------------------------------
    # Eval placeholder
    # ------------------------------------------------------------

    def _run_eval(self):

        print(f"[Eval] step={self.step} — plug in eval dataloader")
