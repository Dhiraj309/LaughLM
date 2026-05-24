"""
LaughLM/training/trainer.py

PMAP production trainer for current LLaMA model.

Runtime:
host batch [global_batch, seq_len]
→ reshape [devices, grad_accum, micro_batch_per_device, seq_len]
→ pmap train_step(axis_name="data")
→ replicated optimizer update
→ save host-unreplicated state
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.parameter_utils import (
    generate_preflight_report,
    estimate_parameters,
)

from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import (
    build_scheduler,
    compute_total_steps,
)

from LaughLM.training.train_step import (
    create_train_step,
    create_eval_step,
)

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


def _unreplicate(tree):
    return jax.tree_util.tree_map(
        lambda x: jax.device_get(x[0]),
        tree,
    )


class Trainer:
    def __init__(
        self,
        config: LaughLMConfig,
        resume_dir: str | None = None,
    ):
        self.config = config

        self.num_devices = jax.local_device_count()
        self.devices = jax.local_devices()

        print(
            f"[trainer] using {self.num_devices} local devices with PMAP",
            flush=True,
        )

        # ====================================================
        # Runtime validation
        # ====================================================

        if self.num_devices <= 0:
            raise RuntimeError(
                "No local JAX devices found."
            )

        if config.runtime.micro_batch_per_device <= 0:
            raise ValueError(
                "runtime.micro_batch_per_device must be > 0"
            )

        if config.runtime.gradient_accumulation <= 0:
            raise ValueError(
                "runtime.gradient_accumulation must be > 0"
            )

        if config.runtime.seq_len <= 0:
            raise ValueError(
                "runtime.seq_len must be > 0"
            )

        if (
            config.runtime.seq_len
            > config.model.max_seq_len
        ):
            raise ValueError(
                f"runtime.seq_len={config.runtime.seq_len} "
                f"exceeds model.max_seq_len={config.model.max_seq_len}"
            )

        if (
            config.parallelism.data_parallel
            != self.num_devices
        ):
            raise ValueError(
                "PMAP requires parallelism.data_parallel "
                "to exactly match jax.local_device_count().\n"
                f"Got config={config.parallelism.data_parallel}, "
                f"devices={self.num_devices}"
            )

        self.rng = create_rng(seed=42)

        generate_preflight_report(
            config,
            num_devices=self.num_devices,
        )

        # ====================================================
        # Checkpoints
        # ====================================================

        ckpt_dir = (
            resume_dir
            or config.runtime.checkpoint_dir
        )

        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=config.runtime.checkpoint_max_to_keep,
        )

        self.checkpoint_interval = (
            config.runtime.checkpoint_interval
        )

        config_path = (
            Path(ckpt_dir) / "config.json"
        )

        if (
            jax.process_index() == 0
            and not config_path.exists()
        ):
            config_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(config_path, "w") as f:
                json.dump(
                    self.config.model_dump(),
                    f,
                    indent=2,
                )

        # ====================================================
        # Model
        # ====================================================

        llama_config = build_llama_config(config)

        self.model = LlamaForCausalLM(
            config=llama_config
        )

        self.grad_accum = (
            config.runtime.gradient_accumulation
        )

        dummy = jnp.zeros(
            (
                config.runtime.micro_batch_per_device,
                config.runtime.seq_len,
            ),
            dtype=jnp.int32,
        )

        variables = self.model.init(
            self.rng.next_key(),
            input_ids=dummy,
            use_cache=False,
            mode="train",
        )

        params = variables["params"]

        # ====================================================
        # Optimizer
        # ====================================================

        self.schedule = build_scheduler(
            config,
            num_devices=self.num_devices,
        )

        self.optimizer = build_optimizer(
            config,
            self.schedule,
        )

        opt_state = self.optimizer.init(params)

        # ====================================================
        # State
        # ====================================================

        state = TrainState(
            params=params,
            opt_state=opt_state,
            step=jnp.array(
                0,
                dtype=jnp.int32,
            ),
            rng_key=self.rng.key,
        )

        restored = self.checkpoints.restore_latest(
            target_state=state,
            config=config,
            num_devices=self.num_devices,
        )

        tokens_per_step_for_resume = (
            config.runtime.seq_len
            * config.runtime.micro_batch_per_device
            * self.num_devices
            * config.runtime.gradient_accumulation
        )

        if restored is not None:
            state, restored_step = restored

            self.start_step = int(state.step)

            self.start_tokens_seen = (
                int(self.start_step)
                * int(tokens_per_step_for_resume)
            )

            print(
                f"[trainer] resumed from step={self.start_step:,} "
                f"tokens={self.start_tokens_seen:,}",
                flush=True,
            )

        else:
            self.start_step = 0
            self.start_tokens_seen = 0

            print(
                "[trainer] fresh run",
                flush=True,
            )

        # ====================================================
        # Replicate state
        # ====================================================

        self.state = jax.device_put_replicated(
            state,
            self.devices,
        )

        # ====================================================
        # Train/eval step
        # ====================================================

        self.train_step = create_train_step(
            model=self.model,
            optimizer=self.optimizer,
            grad_accum=self.grad_accum,
            max_grad_norm=config.optimizer.gradient_clip,
        )

        self.eval_step = create_eval_step(
            model=self.model
        )

        # ====================================================
        # Logger
        # ====================================================

        param_info = estimate_parameters(config)

        self.logger = TrainingLogger(
            config,
            total_params=param_info["total_params"],
            embedding_params=param_info["embedding_params"],
            num_devices=self.num_devices,
        )

        global_batch_size = (
            config.runtime.micro_batch_per_device
            * self.num_devices
        )

        tokens_per_step = (
            config.runtime.seq_len
            * global_batch_size
            * self.grad_accum
        )

        print(
            "[trainer] runtime:\n"
            f"  global_batch={global_batch_size}\n"
            f"  per_device_batch={config.runtime.micro_batch_per_device}\n"
            f"  seq_len={config.runtime.seq_len}\n"
            f"  grad_accum={self.grad_accum}\n"
            f"  tokens_per_step={tokens_per_step:,}",
            flush=True,
        )

    # ========================================================
    # Train loop
    # ========================================================

    def train(self, dataloader: Iterator):
        cfg = self.config

        total_steps = compute_total_steps(
            cfg,
            num_devices=self.num_devices,
        )

        global_batch_size = (
            cfg.runtime.micro_batch_per_device
            * self.num_devices
        )

        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch_size
            * self.grad_accum
        )

        current_step = int(self.start_step)
        host_tokens_seen = int(
            self.start_tokens_seen
        )

        print(
            f"\nTraining for {total_steps:,} optimizer steps with PMAP\n",
            flush=True,
        )

        prefetched_loader = prefetch_to_device(
            iter(dataloader),
            size=8,
        )

        data_iter = iter(prefetched_loader)

        try:
            while True:

                if current_step >= total_steps:
                    break

                step_start = time.perf_counter()

                micro_batches = []

                # ============================================
                # Data loading
                # ============================================

                for _ in range(self.grad_accum):

                    batch = next(data_iter)

                    if not isinstance(batch, np.ndarray):
                        batch = np.asarray(batch)

                    if batch.dtype != np.int32:
                        batch = batch.astype(np.int32)

                    expected_shape = (
                        global_batch_size,
                        cfg.runtime.seq_len,
                    )

                    if batch.shape != expected_shape:
                        raise ValueError(
                            f"Batch shape mismatch: got {batch.shape}, "
                            f"expected {expected_shape}"
                        )

                    micro_batches.append(batch)

                batch = np.stack(
                    micro_batches,
                    axis=0,
                )

                batch = batch.reshape(
                    self.grad_accum,
                    self.num_devices,
                    cfg.runtime.micro_batch_per_device,
                    cfg.runtime.seq_len,
                )

                batch = np.swapaxes(
                    batch,
                    0,
                    1,
                )

                # ============================================
                # Device step
                # ============================================

                self.state, metrics = self.train_step(
                    self.state,
                    jnp.asarray(batch),
                )

                metrics = jax.tree_util.tree_map(
                    lambda x: x.block_until_ready(),
                    metrics,
                )

                self.state.step.block_until_ready()

                step_time = (
                    time.perf_counter()
                    - step_start
                )

                metrics_host = jax.tree_util.tree_map(
                    lambda x: float(
                        jax.device_get(x[0])
                    ),
                    metrics,
                )

                current_step += 1
                host_tokens_seen += tokens_per_step

                lr = _scalar(
                    self.schedule(current_step)
                )

                self.logger.log_metrics(
                    step=current_step,
                    metrics=metrics_host,
                    lr=lr,
                    grad_norm=metrics_host.get(
                        "grad_norm"
                    ),
                    tokens_seen=host_tokens_seen,
                    tokens_in_step=tokens_per_step,
                    step_time=step_time,
                )

                if (
                    current_step
                    % cfg.runtime.log_interval
                    == 0
                ):
                    self.logger.log_step(
                        step=current_step,
                        metrics=metrics_host,
                        lr=lr,
                        grad_norm=metrics_host.get(
                            "grad_norm"
                        ),
                        tokens_seen=host_tokens_seen,
                        tokens_in_step=tokens_per_step,
                        step_time=step_time,
                    )

                # ============================================
                # Checkpoint
                # ============================================

                if (
                    current_step > 0
                    and current_step
                    % self.checkpoint_interval
                    == 0
                ):
                    self.logger.flush()

                    state_to_save = _unreplicate(
                        self.state
                    )

                    metadata = (
                        self.checkpoints.build_metadata_from_config(
                            config=self.config,
                            step=current_step,
                            tokens_processed=host_tokens_seen,
                            num_devices=self.num_devices,
                        )
                    )

                    self.checkpoints.save(
                        step=current_step,
                        state=state_to_save,
                        metadata=metadata,
                    )

                    print(
                        f"[trainer] checkpoint saved "
                        f"step={current_step:,} "
                        f"tokens={host_tokens_seen:,}",
                        flush=True,
                    )

            # ====================================================
            # Final checkpoint
            # ====================================================

            print(
                "[trainer] saving final checkpoint...",
                flush=True,
            )

            self.logger.flush()

            state_to_save = _unreplicate(
                self.state
            )

            final_step = current_step
            final_tokens_seen = host_tokens_seen

            metadata = (
                self.checkpoints.build_metadata_from_config(
                    config=self.config,
                    step=final_step,
                    tokens_processed=final_tokens_seen,
                    num_devices=self.num_devices,
                )
            )

            self.checkpoints.save(
                step=final_step,
                state=state_to_save,
                metadata=metadata,
            )

            self.checkpoints.wait()

            self.logger.log_summary(
                step=final_step,
                tokens_processed=final_tokens_seen,
            )

        finally:
            self.logger.close()

            if hasattr(self.checkpoints, "close"):
                self.checkpoints.close()
            else:
                self.checkpoints.wait()