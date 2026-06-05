"""
LaughLM/training/fsdp_trainer.py

GSPMD/FSDP trainer.

Separate from PMAP Trainer so stable PMAP path remains untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen import partitioning as nn_partitioning

from LaughLM.config.schema import LaughLMConfig
from LaughLM.distributed.mesh import create_mesh
from LaughLM.distributed.sharding import (
    enable_gspmd_constraints,
    set_current_mesh,
    get_logical_axis_rules,
    logical_to_sharding,
    create_input_sharding,
    replicated_sharding,
)
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.parameter_utils import generate_preflight_report, estimate_parameters
from LaughLM.training.train_state import TrainState
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler, compute_total_steps
from LaughLM.training.fsdp_train_step import create_fsdp_train_step
from LaughLM.training.logger import TrainingLogger
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.utils.rng import create_rng
from LaughLM.utils.prefetch import prefetch_to_device


def _scalar(x):
    return float(
        jax.device_get(
            x
        )
    )


def _device_scalar_int(x):
    return int(
        jax.device_get(
            x
        )
    )


class FSDPTrainer:
    def __init__(
        self,
        config: LaughLMConfig,
        resume_dir: str | None = None,
    ):
        backend = getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )

        if backend != "fsdp":
            raise ValueError(
                "FSDPTrainer requires runtime.backend='fsdp' "
                "or temporary alias runtime.backend='gspmd'.\n"
                f"Got raw backend={config.runtime.backend!r}, "
                f"canonical backend={backend!r}."
            )

        if config.runtime.backend == "gspmd":
            print(
                "[fsdp] runtime.backend='gspmd' is deprecated; "
                "using canonical backend='fsdp'.",
                flush=True,
            )

        self.config = config

        self.benchmark_mode = bool(
            getattr(
                config.runtime,
                "benchmark_mode",
                False,
            )
        )

        self.mesh = create_mesh(
            config
        )

        set_current_mesh(
            self.mesh
        )

        self.mesh_axis_sizes = (
            config.spmd.mesh.axis_sizes()
        )

        self.data_replicas = self.mesh_axis_sizes.get(
            "data",
            1,
        )

        self.fsdp_size = self.mesh_axis_sizes.get(
            "fsdp",
            1,
        )

        if self.fsdp_size <= 1:
            raise ValueError(
                "FSDPTrainer requires spmd.mesh fsdp axis > 1."
            )

        enable_gspmd_constraints(
            True
        )

        self.rng = create_rng(
            seed=42
        )

        print(
            "[fsdp] runtime:\n"
            f"  mesh_axes={self.mesh.axis_names}\n"
            f"  mesh_shape={self.mesh.devices.shape}\n"
            f"  data_replicas={self.data_replicas}\n"
            f"  fsdp_size={self.fsdp_size}",
            flush=True,
        )

        print(
            "[fsdp] timing mode:\n"
            f"  benchmark_mode={self.benchmark_mode}",
            flush=True,
        )

        generate_preflight_report(
            config,
            num_devices=self.data_replicas,
        )

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
            Path(ckpt_dir)
            / "config.json"
        )

        if (
            jax.process_index() == 0
            and not config_path.exists()
        ):
            config_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                config_path,
                "w",
            ) as f:
                json.dump(
                    config.model_dump(),
                    f,
                    indent=2,
                )

        llama_config = build_llama_config(
            config
        )

        self.model = LlamaForCausalLM(
            config=llama_config
        )

        self.grad_accum = (
            config.runtime.gradient_accumulation
        )

        self.global_batch_size = (
            config.runtime.micro_batch_per_device
            * self.data_replicas
        )

        self.schedule = build_scheduler(
            config,
            num_devices=self.data_replicas,
        )

        self.optimizer = build_optimizer(
            config,
            self.schedule,
        )

        self.input_sharding = create_input_sharding(
            self.mesh,
            config,
        )

        self.metrics_sharding = replicated_sharding(
            self.mesh,
        )

        self.state = self._init_or_restore_state()

        self.train_step = create_fsdp_train_step(
            model=self.model,
            optimizer=self.optimizer,
            state_sharding=self.state_sharding,
            batch_sharding=self.input_sharding,
            metrics_sharding=self.metrics_sharding,
            grad_accum=self.grad_accum,
            max_grad_norm=config.optimizer.gradient_clip,
            loss_config=config.loss,
        )

        param_info = estimate_parameters(
            config
        )

        self.logger = TrainingLogger(
            config,
            total_params=param_info["total_params"],
            embedding_params=param_info["embedding_params"],
            num_devices=self.data_replicas,
        )

        tokens_per_step = (
            config.runtime.seq_len
            * self.global_batch_size
            * self.grad_accum
        )

        print(
            "[fsdp] training shape:\n"
            f"  global_batch={self.global_batch_size}\n"
            f"  seq_len={config.runtime.seq_len}\n"
            f"  grad_accum={self.grad_accum}\n"
            f"  tokens_per_step={tokens_per_step:,}",
            flush=True,
        )

    def _init_or_restore_state(self):
        cfg = self.config

        dummy_inputs = jax.ShapeDtypeStruct(
            shape=(
                self.global_batch_size,
                cfg.runtime.seq_len,
            ),
            dtype=jnp.int32,
        )

        def init_state_fn(rng):
            variables = self.model.init(
                rng,
                input_ids=dummy_inputs,
                use_cache=False,
                mode="train",
            )

            params = variables["params"]

            opt_state = self.optimizer.init(
                params
            )

            return TrainState(
                params=params,
                opt_state=opt_state,
                step=jnp.asarray(
                    0,
                    dtype=jnp.int32,
                ),
                tokens_processed=jnp.asarray(
                    0,
                    dtype=jnp.int32,
                ),
                rng_key=rng,
            )

        with (
            self.mesh,
            nn_partitioning.axis_rules(
                get_logical_axis_rules(
                    cfg,
                    mesh=self.mesh,
                )
            ),
        ):
            abstract_state = jax.eval_shape(
                init_state_fn,
                self.rng.key,
            )

            logical_state_specs = nn.get_partition_spec(
                abstract_state,
            )

            self.state_sharding = logical_to_sharding(
                logical_state_specs,
                self.mesh,
                cfg,
            )

            init_jit = jax.jit(
                init_state_fn,
                out_shardings=self.state_sharding,
            )

            state = init_jit(
                self.rng.key
            )

        restored = self.checkpoints.restore_latest(
            target_state=state,
            config=cfg,
            num_devices=self.data_replicas,
        )

        if restored is not None:
            state, restored_step = restored

            print(
                f"[fsdp] resumed from step={_device_scalar_int(state.step):,} "
                f"tokens={_device_scalar_int(state.tokens_processed):,}",
                flush=True,
            )

        else:
            print(
                "[fsdp] fresh run",
                flush=True,
            )

        return state

    def train(
        self,
        dataloader: Iterator,
    ):
        cfg = self.config

        total_steps = compute_total_steps(
            cfg,
            num_devices=self.data_replicas,
        )

        tokens_per_step = (
            cfg.runtime.seq_len
            * self.global_batch_size
            * self.grad_accum
        )

        print(
            f"\nTraining for {total_steps:,} optimizer steps with GSPMD/FSDP\n",
            flush=True,
        )

        data_iter = iter(
            prefetch_to_device(
                iter(dataloader),
                size=8,
            )
        )

        try:
            while True:
                step = _device_scalar_int(
                    self.state.step
                )

                if step >= total_steps:
                    break

                total_step_start = time.perf_counter()

                data_wait_time = 0.0
                host_batch_prepare_time = 0.0

                micro_batches = []

                for _ in range(
                    self.grad_accum
                ):
                    data_wait_start = time.perf_counter()

                    batch = next(
                        data_iter
                    )

                    data_wait_time += (
                        time.perf_counter()
                        - data_wait_start
                    )

                    host_prepare_start = time.perf_counter()

                    if not isinstance(
                        batch,
                        np.ndarray,
                    ):
                        batch = np.asarray(
                            batch
                        )

                    if batch.dtype != np.int32:
                        batch = batch.astype(
                            np.int32
                        )

                    expected_shape = (
                        self.global_batch_size,
                        cfg.runtime.seq_len,
                    )

                    if batch.shape != expected_shape:
                        raise ValueError(
                            f"Batch shape mismatch: got {batch.shape}, "
                            f"expected {expected_shape}"
                        )

                    micro_batches.append(
                        batch
                    )

                    host_batch_prepare_time += (
                        time.perf_counter()
                        - host_prepare_start
                    )

                stack_start = time.perf_counter()

                batch = np.stack(
                    micro_batches,
                    axis=0,
                )

                host_batch_prepare_time += (
                    time.perf_counter()
                    - stack_start
                )

                device_put_start = time.perf_counter()

                batch = jax.device_put(
                    batch,
                    self.input_sharding,
                )

                if self.benchmark_mode:
                    # Benchmark mode:
                    # make device_put_time represent real placement time,
                    # not just host dispatch time.
                    batch.block_until_ready()

                device_put_time = (
                    time.perf_counter()
                    - device_put_start
                )

                device_step_start = time.perf_counter()

                self.state, metrics = self.train_step(
                    self.state,
                    batch,
                )

                if self.benchmark_mode:
                    # Benchmark mode:
                    # explicitly separate compiled step completion from
                    # later host metric conversion.
                    metrics = jax.tree_util.tree_map(
                        lambda x: x.block_until_ready(),
                        metrics,
                    )

                    self.state.step.block_until_ready()

                metrics_host = jax.tree_util.tree_map(
                    lambda x: float(
                        jax.device_get(
                            x
                        )
                    ),
                    metrics,
                )

                current_step = _device_scalar_int(
                    self.state.step
                )

                tokens_seen = _device_scalar_int(
                    self.state.tokens_processed
                )

                device_step_time = (
                    time.perf_counter()
                    - device_step_start
                )

                total_step_time = (
                    time.perf_counter()
                    - total_step_start
                )

                expected_tokens_seen = (
                    current_step
                    * tokens_per_step
                )

                if tokens_seen != expected_tokens_seen:
                    raise ValueError(
                        "FSDP token accounting mismatch.\n"
                        f"  step:                 {current_step:,}\n"
                        f"  tokens_seen:          {tokens_seen:,}\n"
                        f"  expected_tokens_seen: {expected_tokens_seen:,}\n"
                        f"  tokens_per_step:      {tokens_per_step:,}"
                    )

                lr = _scalar(
                    self.schedule(
                        current_step
                    )
                )

                timing_breakdown = {
                    "data_wait_time": float(
                        data_wait_time
                    ),
                    "host_batch_prepare_time": float(
                        host_batch_prepare_time
                    ),
                    "device_put_time": float(
                        device_put_time
                    ),
                    "device_step_time": float(
                        device_step_time
                    ),
                    "total_step_time": float(
                        total_step_time
                    ),
                }

                self.logger.log_metrics(
                    step=current_step,
                    metrics=metrics_host,
                    lr=lr,
                    grad_norm=metrics_host.get(
                        "grad_norm"
                    ),
                    tokens_seen=tokens_seen,
                    tokens_in_step=tokens_per_step,
                    step_time=total_step_time,
                    timing_breakdown=timing_breakdown,
                )

                if current_step % cfg.runtime.log_interval == 0:
                    self.logger.log_step(
                        step=current_step,
                        metrics=metrics_host,
                        lr=lr,
                        grad_norm=metrics_host.get(
                            "grad_norm"
                        ),
                        tokens_seen=tokens_seen,
                        tokens_in_step=tokens_per_step,
                        step_time=total_step_time,
                        timing_breakdown=timing_breakdown,
                    )

                if (
                    current_step > 0
                    and current_step % self.checkpoint_interval == 0
                ):
                    self.logger.flush()

                    metadata = self.checkpoints.build_metadata_from_config(
                        config=cfg,
                        step=current_step,
                        tokens_processed=tokens_seen,
                        num_devices=self.data_replicas,
                    )

                    self.checkpoints.save(
                        step=current_step,
                        state=self.state,
                        metadata=metadata,
                    )

                    print(
                        f"[fsdp] checkpoint saved step={current_step:,}",
                        flush=True,
                    )

            print(
                "[fsdp] saving final checkpoint...",
                flush=True,
            )

            self.logger.flush()

            final_step = _device_scalar_int(
                self.state.step
            )

            tokens_seen = _device_scalar_int(
                self.state.tokens_processed
            )

            metadata = self.checkpoints.build_metadata_from_config(
                config=cfg,
                step=final_step,
                tokens_processed=tokens_seen,
                num_devices=self.data_replicas,
            )

            self.checkpoints.save(
                step=final_step,
                state=self.state,
                metadata=metadata,
            )

            self.checkpoints.wait()

            self.logger.log_summary(
                step=final_step,
                tokens_processed=tokens_seen,
            )

        finally:
            self.logger.close()

            if hasattr(
                self.checkpoints,
                "close",
            ):
                self.checkpoints.close()

            else:
                self.checkpoints.wait()
