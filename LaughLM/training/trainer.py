import json
import time
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
from LaughLM.training.train_state import TrainState
from LaughLM.utils.rng import create_rng
from LaughLM.utils.prefetch import prefetch_to_device


def _scalar(x):
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

    def __init__(self, config: LaughLMConfig, resume_dir: str | None = None):

        self.config = config
        self.num_devices = jax.device_count()
        print(f"[trainer] Using {self.num_devices} devices")

        self.rng = create_rng(seed=42)
        generate_preflight_report(config)

        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (config.runtime.micro_batch_per_device, config.runtime.seq_len),
            dtype=jnp.int32,
        )

        params = self.model.init(self.rng.next_key(), dummy)["params"]

        self.schedule = build_scheduler(config)
        self.optimizer = build_optimizer(config, self.schedule)

        opt_state = self.optimizer.init(params)

        # replicate across devices
        params = jax.device_put_replicated(params, jax.devices())
        opt_state = jax.device_put_replicated(opt_state, jax.devices())

        self.grad_accum = config.runtime.gradient_accumulation

        self.train_step = create_train_step(
            self.model,
            self.optimizer,
            self.grad_accum,
        )

        self.eval_step = create_eval_step(self.model)

        self.state = TrainState(
            params=params,
            opt_state=opt_state,
            step=0,
            tokens_processed=0,
            rng_key=self.rng.key,
        )

        # ✅ params for MFU
        param_info   = estimate_parameters(config)
        total_params = param_info["total_params"]
        emb_params   = param_info["embedding_params"]

        self.logger = TrainingLogger(
            config,
            total_params=total_params,
            embedding_params=emb_params,
        )

        ckpt_dir = resume_dir or config.runtime.checkpoint_dir

        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=config.runtime.checkpoint_max_to_keep,
        )

        self.checkpoint_interval = config.runtime.checkpoint_interval

        config_path = Path(ckpt_dir) / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

        restored = self.checkpoints.restore_latest(self.state)
        if restored is not None:
            self.state, step = restored

            self.state = TrainState(
                params=jax.device_put_replicated(self.state.params, jax.devices()),
                opt_state=jax.device_put_replicated(self.state.opt_state, jax.devices()),
                step=self.state.step,
                tokens_processed=self.state.tokens_processed,
                rng_key=self.state.rng_key,
            )

            self.rng._key = self.state.rng_key
            print(f"[trainer] resumed from step {self.state.step}")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    def train(self, dataloader: Iterator):

        cfg = self.config
        total_steps = compute_total_steps(cfg)

        # ============================================================
        # ✅ FIX: CORRECT TOKEN ACCOUNTING (NO DOUBLE COUNTING)
        # ============================================================

        global_batch_size = (
            cfg.runtime.micro_batch_per_device
            * self.num_devices
        )

        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch_size
            * cfg.runtime.gradient_accumulation
        )

        # ✅ DEBUG (remove later)
        print(f"[debug] global_batch_size = {global_batch_size}")
        print(f"[debug] tokens_per_step = {tokens_per_step:,}")

        print("\n" + "=" * 60)
        print(f"Training for {total_steps:,} optimizer steps")
        print(f"Effective tokens per step: {tokens_per_step:,}")
        print("=" * 60 + "\n")

        prefetched_loader = prefetch_to_device(iter(dataloader), size=8)
        data_iter = iter(prefetched_loader)

        for _ in range(total_steps):

            # ✅ start timing EARLY (correct placement)
            step_start = time.time()

            micro_batches = []
            for _ in range(self.grad_accum):
                batch = next(data_iter)
                batch = jnp.asarray(batch, dtype=jnp.int32)
                micro_batches.append(batch)

            # shape: (grad_accum, devices, micro_batch, seq)
            batch = jnp.stack(micro_batches)

            # shape: (devices, grad_accum, micro_batch, seq)
            batch = jnp.swapaxes(batch, 0, 1)

            # ✅ DEBUG SHAPE (run once)
            if self.state.step == 0:
                print(f"[debug] batch shape: {batch.shape}")

            new_params, new_opt_state, metrics = self.train_step(
                self.state.params,
                self.state.opt_state,
                batch,
            )

            # ============================================================
            # ✅ CORRECT SYNC + METRICS EXTRACTION
            # ============================================================

            metrics = {
                k: float(jax.device_get(v).mean())
                for k, v in metrics.items()
            }

            # ✅ TRUE STEP TIME
            step_time = time.time() - step_start

            new_step = self.state.step + 1
            new_tokens = self.state.tokens_processed + tokens_per_step

            self.state = TrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=new_step,
                tokens_processed=new_tokens,
                rng_key=self.rng.key,
            )

            if self.state.step % cfg.runtime.log_interval == 0:
                lr = _scalar(self.schedule(self.state.step))

                self.logger.log_step(
                    step=self.state.step,
                    metrics=metrics,
                    lr=lr,
                    grad_norm=None,
                    tokens_seen=self.state.tokens_processed,
                    tokens_in_step=tokens_per_step,
                    step_time=step_time,
                )

            if self.state.step % self.checkpoint_interval == 0:
                state_to_save = TrainState(
                    params=jax.tree_util.tree_map(lambda x: x[0], self.state.params),
                    opt_state=jax.tree_util.tree_map(lambda x: x[0], self.state.opt_state),
                    step=self.state.step,
                    tokens_processed=self.state.tokens_processed,
                    rng_key=self.state.rng_key,
                )

                self.checkpoints.save(self.state.step, state_to_save)

        # Final checkpoint
        state_to_save = TrainState(
            params=jax.tree_util.tree_map(lambda x: x[0], self.state.params),
            opt_state=jax.tree_util.tree_map(lambda x: x[0], self.state.opt_state),
            step=self.state.step,
            tokens_processed=self.state.tokens_processed,
            rng_key=self.state.rng_key,
        )

        self.checkpoints.save(self.state.step, state_to_save)

        # ✅ ensure async writes finish
        self.checkpoints.wait()

        self.logger.log_summary(
            self.state.step,
            self.state.tokens_processed,
        )

    def _run_eval(self):
        print(f"[Eval] step={self.state.step} — plug in eval dataloader")