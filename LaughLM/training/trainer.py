import json
from pathlib import Path
from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

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
        self.rng = create_rng(seed=42)

        generate_preflight_report(config)

        # ----------------------------------------------------------------
        # Mesh
        # ----------------------------------------------------------------
        n_devices = jax.device_count()
        dp = config.parallelism.data_parallel
        assert dp == n_devices, (
            f"data_parallel={dp} but jax.device_count()={n_devices}."
        )
        self.mesh = Mesh(np.array(jax.devices()), axis_names=('data',))

        self._replicated = NamedSharding(self.mesh, P())
        self._batch_shard = NamedSharding(self.mesh, P(None, 'data', None))

        # ----------------------------------------------------------------
        # Model
        # ----------------------------------------------------------------
        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (config.runtime.micro_batch_per_device, config.runtime.seq_len),
            dtype=jnp.int32,
        )

        params = self.model.init(self.rng.next_key(), dummy)["params"]

        self.schedule = build_scheduler(config)
        self.optimizer = build_optimizer(config, self.schedule)
        opt_state = self.optimizer.init(params)

        params = jax.device_put(params, self._replicated)
        opt_state = jax.device_put(opt_state, self._replicated)

        # ----------------------------------------------------------------
        # Train / eval steps
        # ----------------------------------------------------------------
        self.grad_accum = config.runtime.gradient_accumulation

        self.train_step = create_train_step(
            self.model,
            self.optimizer,
            self.grad_accum,
            mesh=self.mesh,
        )
        self.eval_step = create_eval_step(self.model)

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        self.state = TrainState(
            params=params,
            opt_state=opt_state,
            step=0,
            tokens_processed=0,
            rng_key=self.rng.key,
        )

        total_params = estimate_parameters(config)["total_params"]
        self.logger = TrainingLogger(config, total_params=total_params)

        ckpt_dir = resume_dir or config.runtime.checkpoint_dir
        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=config.runtime.checkpoint_max_to_keep
        )
        self.checkpoint_interval = config.runtime.checkpoint_interval

        config_path = Path(ckpt_dir) / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

        restored = self.checkpoints.restore_latest(self.state)
        if restored is not None:
            self.state, _ = restored
            self.rng._key = self.state.rng_key
            print(f"[trainer] resumed from step {self.state.step}")

    # ----------------------------------------------------------------
    # Evaluation (🔥 FIXED)
    # ----------------------------------------------------------------
    def evaluate(self, val_dataset, num_batches=50):

        total_loss = 0.0
        count = 0

        val_iter = iter(val_dataset)

        for _ in range(num_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            # ------------------------------------------------------------
            # FIX: match training sharding shape
            # ------------------------------------------------------------
            batch = jnp.asarray(batch, dtype=jnp.int32)

            # add grad_accum dim → [1, B, T]
            batch = jnp.expand_dims(batch, axis=0)

            batch = jax.device_put(batch, self._batch_shard)

            # remove grad_accum dim for eval_step
            metrics = self.eval_step(self.state.params, batch[0])
            metrics = jax.device_get(metrics)

            total_loss += float(metrics["loss"])
            count += 1

        if count == 0:
            return None

        return total_loss / count

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    def train(self, dataloader: Iterator, val_dataset: Iterator | None = None):

        cfg = self.config
        total_steps = compute_total_steps(cfg)

        tokens_per_step = (
            cfg.runtime.seq_len
            * cfg.runtime.micro_batch_per_device
            * cfg.parallelism.data_parallel
            * cfg.runtime.gradient_accumulation
        )

        print("\n" + "=" * 60)
        print(f"Training for {total_steps:,} steps")
        print(f"Tokens per step: {tokens_per_step:,}")
        print("=" * 60 + "\n")

        prefetched_loader = prefetch_to_device(iter(dataloader), size=8)
        data_iter = iter(prefetched_loader)

        with self.mesh:

            start_step = self.state.step
            remaining_steps = total_steps - start_step

            for _ in range(remaining_steps):

                # --------------------------------------------------------
                # Load batch
                # --------------------------------------------------------
                micro_batches = []
                for _ in range(self.grad_accum):
                    batch = next(data_iter)
                    micro_batches.append(jnp.asarray(batch, dtype=jnp.int32))

                batch = jnp.stack(micro_batches)
                batch = jax.device_put(batch, self._batch_shard)

                # --------------------------------------------------------
                # Train step
                # --------------------------------------------------------
                new_params, new_opt_state, metrics = self.train_step(
                    self.state.params,
                    self.state.opt_state,
                    batch,
                )

                jax.block_until_ready(metrics)

                # --------------------------------------------------------
                # Update state
                # --------------------------------------------------------
                new_step = self.state.step + 1
                new_tokens = self.state.tokens_processed + tokens_per_step

                self.state = TrainState(
                    params=new_params,
                    opt_state=new_opt_state,
                    step=new_step,
                    tokens_processed=new_tokens,
                    rng_key=self.rng.key,
                )

                # --------------------------------------------------------
                # Logging
                # --------------------------------------------------------
                if self.state.step % cfg.runtime.log_interval == 0:

                    metrics_host = jax.device_get(metrics)
                    lr = _scalar(self.schedule(self.state.step))

                    self.logger.log_step(
                        step=self.state.step,
                        metrics={k: _scalar(v) for k, v in metrics_host.items()},
                        lr=lr,
                        grad_norm=None,
                        tokens_seen=self.state.tokens_processed,
                    )

                # --------------------------------------------------------
                # Validation
                # --------------------------------------------------------
                if val_dataset is not None and self.state.step % cfg.runtime.eval_interval == 0:

                    val_loss = self.evaluate(val_dataset)

                    if val_loss is not None:
                        print(f"[VAL] step={self.state.step} loss={val_loss:.4f}")

                # --------------------------------------------------------
                # Checkpoint
                # --------------------------------------------------------
                if self.state.step % self.checkpoint_interval == 0:
                    self.checkpoints.save(self.state.step, self.state)

        # --------------------------------------------------------
        # Final checkpoint
        # --------------------------------------------------------
        self.checkpoints.save(self.state.step, self.state, wait=True)
        self.checkpoints.close()

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------
        self.logger.log_summary(
            self.state.step,
            self.state.tokens_processed,
        )