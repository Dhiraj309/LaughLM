"""
LaughLM/training/trainer.py

FINAL FIXED VERSION (2026)

Fixes:
──────────────────────────────────────────────
1. Guaranteed correct tokens_per_step
2. Strict batch size validation (prevents silent bugs)
3. Uses state.step instead of blind loop
4. Safer device → host extraction
5. Perfect alignment with train_step token accounting
"""

import json
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.parameter_utils import (
    generate_preflight_report,
    estimate_parameters,
)
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler, compute_total_steps
from LaughLM.training.train_step import create_train_step, create_eval_step
from LaughLM.training.logger import TrainingLogger
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.train_state import TrainState
from LaughLM.utils.rng import create_rng
from LaughLM.utils.prefetch import prefetch_to_device


def _scalar(x):
    try:
        return float(jax.device_get(x))
    except Exception:
        return float("nan")


class Trainer:

    def __init__(self, config: LaughLMConfig, resume_dir: str | None = None):

        self.config = config
        self.num_devices = jax.device_count()
        self.devices = jax.devices()

        print(f"[trainer] Using {self.num_devices} devices (pmap)")

        self.rng = create_rng(seed=42)
        generate_preflight_report(config)

        # ─────────────────────────────────────────────
        # Model init
        # ─────────────────────────────────────────────

        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (config.runtime.micro_batch_per_device, config.runtime.seq_len),
            dtype=jnp.int32,
        )

        params = self.model.init(self.rng.next_key(), dummy)["params"]

        self.schedule = build_scheduler(config)
        self.optimizer = build_optimizer(config, self.schedule)

        opt_state = self.optimizer.init(params)

        state = TrainState(
            params=params,
            opt_state=opt_state,
            step=0,
            tokens_processed=0,
            rng_key=self.rng.key,
        )

        self.state = jax.device_put_replicated(state, self.devices)

        self.grad_accum = config.runtime.gradient_accumulation

        # ─────────────────────────────────────────────
        # Train step
        # ─────────────────────────────────────────────

        self.train_step = create_train_step(
            self.model,
            self.optimizer,
            self.grad_accum,
            max_grad_norm=config.optimizer.gradient_clip,
        )

        self.eval_step = create_eval_step(self.model)

        # ─────────────────────────────────────────────
        # Logging
        # ─────────────────────────────────────────────

        param_info = estimate_parameters(config)

        self.logger = TrainingLogger(
            config,
            total_params=param_info["total_params"],
            embedding_params=param_info["embedding_params"],
        )

        # ─────────────────────────────────────────────
        # Checkpoints
        # ─────────────────────────────────────────────

        ckpt_dir = resume_dir or config.runtime.checkpoint_dir

        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=config.runtime.checkpoint_max_to_keep,
        )

        self.checkpoint_interval = config.runtime.checkpoint_interval

        config_path = Path(ckpt_dir) / "config.json"
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

    # ─────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────

    def train(self, dataloader: Iterator):

        cfg = self.config
        total_steps = compute_total_steps(cfg)

        # ✅ TRUE GLOBAL BATCH
        global_batch_size = (
            cfg.runtime.micro_batch_per_device * self.num_devices
        )

        # ✅ SINGLE SOURCE OF TRUTH
        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch_size
            * self.grad_accum
        )

        print(f"\n{'=' * 60}")
        print(f"Training for {total_steps:,} optimizer steps (pmap)")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Global batch: {global_batch_size} seqs")
        print(f"Grad accum: {self.grad_accum}")
        print(f"{'=' * 60}\n")

        prefetched_loader = prefetch_to_device(iter(dataloader), size=8)
        data_iter = iter(prefetched_loader)

        # ✅ USE STATE STEP (important for resume correctness)
        while True:

            state_host = jax.tree_util.tree_map(lambda x: x[0], self.state)
            step = int(state_host.step)

            if step >= total_steps:
                break

            step_start = time.time()

            # ─────────────────────────────────────────
            # Load microbatches
            # ─────────────────────────────────────────

            micro_batches = []
            for _ in range(self.grad_accum):
                batch = next(data_iter)

                if batch.dtype != jnp.int32:
                    batch = batch.astype(jnp.int32)

                # 🔥 CRITICAL SHAPE CHECK
                expected = global_batch_size
                assert batch.shape[0] == expected, (
                    f"Batch mismatch: got {batch.shape[0]}, expected {expected}"
                )

                micro_batches.append(batch)

            batch = jnp.stack(micro_batches)

            # reshape → (grad_accum, devices, micro_batch_per_device, seq)
            batch = batch.reshape(
                self.grad_accum,
                self.num_devices,
                cfg.runtime.micro_batch_per_device,
                cfg.runtime.seq_len,
            )

            # → (devices, grad_accum, micro_batch_per_device, seq)
            batch = jnp.swapaxes(batch, 0, 1)

            # ─────────────────────────────────────────
            # Train step
            # ─────────────────────────────────────────

            self.state, metrics = self.train_step(self.state, batch)

            metrics = jax.tree_util.tree_map(
                lambda x: float(jax.device_get(x[0])),
                metrics,
            )

            state_host = jax.tree_util.tree_map(
                lambda x: x[0],
                self.state,
            )

            step_time = time.time() - step_start

            # ─────────────────────────────────────────
            # Logging
            # ─────────────────────────────────────────

            if step % cfg.runtime.log_interval == 0:

                lr = _scalar(self.schedule(step))

                self.logger.log_step(
                    step=step,
                    metrics=metrics,
                    lr=lr,
                    grad_norm=metrics.get("grad_norm"),
                    tokens_seen=int(state_host.tokens_processed),
                    tokens_in_step=tokens_per_step,
                    step_time=step_time,
                )

            # ─────────────────────────────────────────
            # Checkpoint
            # ─────────────────────────────────────────

            if step % self.checkpoint_interval == 0:

                state_to_save = jax.tree_util.tree_map(
                    lambda x: x[0],
                    self.state,
                )

                self.checkpoints.save(step, state_to_save)

        # ─────────────────────────────────────────────
        # Final save
        # ─────────────────────────────────────────────

        state_to_save = jax.tree_util.tree_map(
            lambda x: x[0],
            self.state,
        )

        self.checkpoints.save(step, state_to_save)

        self.checkpoints.wait()

        self.logger.log_summary(
            step,
            int(state_host.tokens_processed),
        )