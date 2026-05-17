
"""
LaughLM/training/trainer.py

Mesh-native SPMD trainer for LaughLM.

Architecture:
──────────────────────────────────────────────
- Pure GSPMD / NamedSharding training
- No pmap
- No replicated TrainState
- Mesh-native parameter initialization
- Sharded optimizer state
- Explicit input batch sharding
- Resume-safe checkpoints
- CPU-side async prefetch
- Static-shape training loop
- Compatible with:
    - FSDP
    - tensor parallelism
    - sequence parallelism
    - scan-over-layers
    - remat

References:
──────────────────────────────────────────────
- MaxText
- Levanter
- Pax
- T5X
"""

import json
import time

from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from flax.linen import (
    partitioning as nn_partitioning,
)

from LaughLM.config.schema import (
    LaughLMConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.model.llama.config_factory import (
    build_llama_config,
)

from LaughLM.model.parameter_utils import (
    generate_preflight_report,
    estimate_parameters,
)

from LaughLM.training.optimizer import (
    build_optimizer,
)

from LaughLM.training.scheduler import (
    build_scheduler,
    compute_total_steps,
)

from LaughLM.training.train_step import (
    create_train_step,
    create_eval_step,
)

from LaughLM.training.logger import (
    TrainingLogger,
)

from LaughLM.training.checkpoint import (
    CheckpointManager,
)

from LaughLM.training.train_state import (
    TrainState,
)

from LaughLM.utils.rng import (
    create_rng,
)

from LaughLM.utils.prefetch import (
    prefetch_to_device,
)

from LaughLM.distributed.mesh import (
    create_mesh,
)

from LaughLM.distributed.state import (
    create_abstract_state,
    create_sharded_state,
)

from LaughLM.distributed.sharding import (
    get_logical_axis_rules,
    create_input_sharding,
)


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def _scalar(x):

    try:
        return float(jax.device_get(x))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        config: LaughLMConfig,
        resume_dir: str | None = None,
    ):

        self.config = config

        # --------------------------------------------------
        # Devices + Mesh
        # --------------------------------------------------

        self.num_devices = jax.device_count()

        self.mesh = create_mesh(config)

        print(
            f"[trainer] Using "
            f"{self.num_devices} devices "
            f"with mesh axes="
            f"{self.mesh.axis_names}"
        )

        # --------------------------------------------------
        # Explicit input sharding
        # --------------------------------------------------

        self.input_sharding = (
            create_input_sharding(
                self.mesh
            )
        )

        # --------------------------------------------------
        # RNG
        # --------------------------------------------------

        self.rng = create_rng(seed=42)

        # --------------------------------------------------
        # Reports
        # --------------------------------------------------

        generate_preflight_report(
            config,
            num_devices=self.num_devices,
        )

        # --------------------------------------------------
        # Checkpoints
        # --------------------------------------------------

        ckpt_dir = (
            resume_dir
            or config.runtime.checkpoint_dir
        )

        self.checkpoints = CheckpointManager(
            ckpt_dir,
            max_to_keep=(
                config.runtime
                .checkpoint_max_to_keep
            ),
        )

        self.checkpoint_interval = (
            config.runtime
            .checkpoint_interval
        )

        config_path = (
            Path(ckpt_dir)
            / "config.json"
        )

        if not config_path.exists():

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

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        llama_config = build_llama_config(
            config
        )

        self.model = LlamaForCausalLM(
            config=llama_config
        )

        # --------------------------------------------------
        # Runtime params
        # --------------------------------------------------

        self.grad_accum = (
            config.runtime
            .gradient_accumulation
        )

        # --------------------------------------------------
        # GLOBAL input shape
        #
        # IMPORTANT:
        # GSPMD initializes with global shapes,
        # not per-device shapes.
        # --------------------------------------------------

        global_batch_size = (
            config.runtime
            .micro_batch_per_device
            * self.num_devices
        )

        input_shape = (
            global_batch_size,
            config.runtime.seq_len,
        )

        # --------------------------------------------------
        # Abstract state + shardings
        # --------------------------------------------------

        (
            self.abstract_state,
            self.logical_specs,
            self.shardings,
        ) = create_abstract_state(
            self.model,
            config,
            self.mesh,
            self.rng.next_key(),
            input_shape,
        )

        # --------------------------------------------------
        # Materialize sharded params
        # --------------------------------------------------

        variables = create_sharded_state(
            self.model,
            config,
            self.mesh,
            self.rng.next_key(),
            input_shape,
        )

        params = variables["params"]

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        self.schedule = build_scheduler(
            config,
            num_devices=self.num_devices,
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = build_optimizer(
            config,
            self.schedule,
        )

        opt_state = self.optimizer.init(
            params
        )

        # --------------------------------------------------
        # Initial train state
        # --------------------------------------------------

        state = TrainState(
            params=params,
            opt_state=opt_state,
            step=0,
            tokens_processed=0,
            rng_key=self.rng.key,
        )

        # --------------------------------------------------
        # Restore checkpoint
        # --------------------------------------------------

        restored = (
            self.checkpoints
            .restore_latest(
                target_state=state
            )
        )

        if restored is not None:

            state, restored_step = restored

            print(
                f"[trainer] resumed from "
                f"step={int(state.step):,} "
                f"tokens="
                f"{int(state.tokens_processed):,}"
            )

        else:

            print(
                "[trainer] starting "
                "fresh training run"
            )

        self.state = state

        # --------------------------------------------------
        # Train / Eval step
        # --------------------------------------------------

        with (
            jax.set_mesh(self.mesh),
            nn_partitioning.axis_rules(
                get_logical_axis_rules(config)
            ),
        ):

            self.train_step = create_train_step(
                model=self.model,
                optimizer=self.optimizer,
                config=config,
                mesh=self.mesh,
                state_shardings=self.shardings,
                data_sharding=self.input_sharding,
                grad_accum=self.grad_accum,
                max_grad_norm=(
                    config.optimizer
                    .gradient_clip
                ),
            )

            self.eval_step = create_eval_step(
                model=self.model,
                config=config,
                mesh=self.mesh,
                data_sharding=self.input_sharding,
            )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        param_info = estimate_parameters(
            config
        )

        self.logger = TrainingLogger(
            config,
            total_params=(
                param_info["total_params"]
            ),
            embedding_params=(
                param_info[
                    "embedding_params"
                ]
            ),
            num_devices=self.num_devices,
        )

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────

    def train(
        self,
        dataloader: Iterator,
    ):

        cfg = self.config

        # --------------------------------------------------
        # Total steps
        # --------------------------------------------------

        total_steps = compute_total_steps(
            cfg,
            num_devices=self.num_devices,
        )

        # --------------------------------------------------
        # Global batch size
        # --------------------------------------------------

        global_batch_size = (
            cfg.runtime
            .micro_batch_per_device
            * self.num_devices
        )

        # --------------------------------------------------
        # Tokens per optimizer step
        # --------------------------------------------------

        tokens_per_step = (
            cfg.runtime.seq_len
            * global_batch_size
            * self.grad_accum
        )

        print(f"\n{'=' * 60}")

        print(
            f"Training for "
            f"{total_steps:,} "
            f"optimizer steps "
            f"(GSPMD)"
        )

        print(
            f"Tokens per step: "
            f"{tokens_per_step:,}"
        )

        print(
            f"Global batch: "
            f"{global_batch_size} seqs"
        )

        print(
            f"Grad accum: "
            f"{self.grad_accum}"
        )

        print(f"{'=' * 60}\n")

        # --------------------------------------------------
        # CPU-side async prefetch
        # --------------------------------------------------

        prefetched_loader = (
            prefetch_to_device(
                iter(dataloader),
                size=8,
            )
        )

        data_iter = iter(
            prefetched_loader
        )

        # --------------------------------------------------
        # Resume-safe training loop
        # --------------------------------------------------

        while True:

            step = int(
                jax.device_get(
                    self.state.step
                )
            )

            if step >= total_steps:
                break

            step_start = time.time()

            # ------------------------------------------
            # Load grad accumulation microbatches
            # ------------------------------------------

            micro_batches = []

            for _ in range(
                self.grad_accum
            ):

                batch = next(data_iter)

                if not isinstance(
                    batch,
                    np.ndarray,
                ):
                    batch = np.asarray(batch)

                if batch.dtype != np.int32:

                    batch = batch.astype(
                        np.int32
                    )

                expected = (
                    global_batch_size
                )

                assert (
                    batch.shape[0]
                    == expected
                ), (
                    f"Batch mismatch: "
                    f"got {batch.shape[0]}, "
                    f"expected {expected}"
                )

                micro_batches.append(
                    batch
                )

            # ------------------------------------------
            # Stack:
            #
            # [grad_accum, global_batch, seq]
            # ------------------------------------------

            batch = np.stack(
                micro_batches
            )

            # ------------------------------------------
            # Explicit input sharding
            #
            # IMPORTANT:
            # No implicit placement.
            # No hidden resharding.
            # Canonical GSPMD semantics.
            # ------------------------------------------

            batch = jax.device_put(
                batch,
                self.input_sharding,
            )

            # ------------------------------------------
            # Train step
            # ------------------------------------------

            self.state, metrics = (
                self.train_step(
                    self.state,
                    batch,
                )
            )

            metrics = (
                jax.tree_util.tree_map(
                    lambda x: float(
                        jax.device_get(x)
                    ),
                    metrics,
                )
            )

            step_time = (
                time.time()
                - step_start
            )

            # ------------------------------------------
            # Scalar extraction ONLY
            #
            # IMPORTANT:
            # Never device_get full state
            # inside hot loop.
            # ------------------------------------------

            tokens_seen = int(
                jax.device_get(
                    self.state.tokens_processed
                )
            )

            # ------------------------------------------
            # Logging
            # ------------------------------------------

            if (
                step
                % cfg.runtime.log_interval
                == 0
            ):

                lr = _scalar(
                    self.schedule(step)
                )

                self.logger.log_step(
                    step=step,
                    metrics=metrics,
                    lr=lr,
                    grad_norm=metrics.get(
                        "grad_norm"
                    ),
                    tokens_seen=tokens_seen,
                    tokens_in_step=(
                        tokens_per_step
                    ),
                    step_time=step_time,
                )

            # ------------------------------------------
            # Checkpoint
            # ------------------------------------------

            if (
                step > 0
                and step
                % self.checkpoint_interval
                == 0
            ):

                state_to_save = (
                    jax.tree_util.tree_map(
                        jax.device_get,
                        self.state,
                    )
                )

                self.checkpoints.save(
                    step,
                    state_to_save,
                )

        # --------------------------------------------------
        # Final checkpoint
        # --------------------------------------------------

        state_to_save = (
            jax.tree_util.tree_map(
                jax.device_get,
                self.state,
            )
        )

        final_step = int(
            state_to_save.step
        )

        self.checkpoints.save(
            final_step,
            state_to_save,
        )

        self.checkpoints.wait()

        self.logger.log_summary(
            final_step,
            int(
                state_to_save
                .tokens_processed
            ),
        )
