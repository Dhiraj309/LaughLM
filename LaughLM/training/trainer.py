"""
LaughLM/training/trainer.py

Mesh-native SPMD trainer for LaughLM.

Frontier-grade fixes (2026):
────────────────────────────────────────────
1. Correct TrainState sharding tree
2. Correct param subtree extraction
3. Proper mesh-native compilation
4. Real synchronous timing (accurate tok/s + MFU)
5. No async timing inflation
6. Stable checkpoint semantics
7. Exact structure match between:
      state
      abstract state
      shardings
8. Safe metrics synchronization
9. TRUE optimizer-state sharding
10. FSDP-ready optimizer partition propagation
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

from jax.sharding import (
    NamedSharding,
    PartitionSpec as P,
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


# ============================================================
# Utils
# ============================================================

def _scalar(x):

    try:
        return float(jax.device_get(x))

    except Exception:
        return float("nan")


# ============================================================
# Helper
# ============================================================

def _create_optimizer_shardings(
    opt_state,
    param_shardings,
    replicated,
):
    """
    Create optimizer-state shardings.

    IMPORTANT
    ─────────────────────────────────────────
    Adam moments MUST shard identically
    to parameters for real FSDP memory scaling.

    Falls back to replicated for scalar
    optimizer metadata.
    """

    flat_param_shardings = (
        jax.tree_util.tree_leaves(
            param_shardings
        )
    )

    param_iter = iter(
        flat_param_shardings
    )

    def map_leaf(x):

        #
        # Scalar metadata
        #

        if (
            np.isscalar(x)
            or not hasattr(x, "shape")
        ):
            return replicated

        #
        # Match parameter sharding
        #

        try:
            return next(param_iter)

        except StopIteration:
            return replicated

    return jax.tree_util.tree_map(
        map_leaf,
        opt_state,
    )


# ============================================================
# Trainer
# ============================================================

class Trainer:

    def __init__(
        self,
        config: LaughLMConfig,
        resume_dir: str | None = None,
    ):

        self.config = config

        # --------------------------------------------------
        # Devices + mesh
        # --------------------------------------------------

        self.num_devices = jax.device_count()

        self.mesh = create_mesh(config)

        print(
            f"[trainer] using "
            f"{self.num_devices} devices "
            f"mesh={self.mesh.axis_names}"
        )

        # --------------------------------------------------
        # Input sharding
        # --------------------------------------------------

        self.input_sharding = create_input_sharding(
            self.mesh,
            config,
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
                config.runtime.checkpoint_max_to_keep
            ),
        )

        self.checkpoint_interval = (
            config.runtime.checkpoint_interval
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

        llama_config = build_llama_config(config)

        self.model = LlamaForCausalLM(
            config=llama_config
        )

        # --------------------------------------------------
        # Runtime params
        # --------------------------------------------------

        self.grad_accum = (
            config.runtime.gradient_accumulation
        )

        global_batch_size = (
            config.runtime.micro_batch_per_device
            * self.num_devices
        )

        input_shape = (
            global_batch_size,
            config.runtime.seq_len,
        )

        # --------------------------------------------------
        # Abstract state metadata
        # --------------------------------------------------

        (
            _abstract_variables,
            self.logical_specs,
            full_shardings,
        ) = create_abstract_state(
            model=self.model,
            config=config,
            mesh=self.mesh,
            rng=self.rng.next_key(),
            input_shape=input_shape,
        )

        #
        # full_shardings matches:
        #
        # {
        #   "params": ...
        # }
        #

        param_shardings = (
            full_shardings["params"]
        )

        # --------------------------------------------------
        # Materialize params
        # --------------------------------------------------

        variables = create_sharded_state(
            model=self.model,
            config=config,
            mesh=self.mesh,
            rng=self.rng.next_key(),
            input_shape=input_shape,
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
        # Train state
        # --------------------------------------------------

        state = TrainState(
            params=params,

            opt_state=opt_state,

            step=jnp.array(
                0,
                dtype=jnp.int32,
            ),

            tokens_processed=jnp.array(
                0,
                dtype=jnp.int32,
            ),

            rng_key=self.rng.key,
        )

        # --------------------------------------------------
        # TrainState shardings
        # --------------------------------------------------

        replicated = NamedSharding(
            self.mesh,
            P(),
        )

        #
        # IMPORTANT
        #
        # Optimizer states MUST shard
        # with parameters.
        #
        # Otherwise:
        #
        # - Adam moments replicate
        # - memory explodes on TPU
        # - FSDP becomes fake
        #

        opt_state_shardings = (
            _create_optimizer_shardings(
                opt_state=opt_state,
                param_shardings=param_shardings,
                replicated=replicated,
            )
        )

        self.state_shardings = TrainState(
            params=param_shardings,

            opt_state=opt_state_shardings,

            step=replicated,

            tokens_processed=replicated,

            rng_key=replicated,
        )

        # --------------------------------------------------
        # Restore checkpoint
        # --------------------------------------------------

        restored = self.checkpoints.restore_latest(
            target_state=state
        )

        if restored is not None:

            state, restored_step = restored

            print(
                f"[trainer] resumed "
                f"from step={restored_step:,}"
            )

        else:

            print(
                "[trainer] fresh run"
            )

        self.state = state

        # --------------------------------------------------
        # Train / eval step
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

                state_shardings=(
                    self.state_shardings
                ),

                data_sharding=self.input_sharding,

                grad_accum=self.grad_accum,

                max_grad_norm=(
                    config.optimizer.gradient_clip
                ),
            )

            self.eval_step = create_eval_step(
                model=self.model,

                config=config,

                mesh=self.mesh,

                state_shardings=(
                    self.state_shardings
                ),

                data_sharding=self.input_sharding,
            )

        # --------------------------------------------------
        # Explicit compile warmup
        # --------------------------------------------------

        print(
            "[trainer] compiling train step..."
        )

        shaped_batch = jax.ShapeDtypeStruct(
            (
                self.grad_accum,
                global_batch_size,
                config.runtime.seq_len,
            ),
            jnp.int32,
        )

        abstract_train_state = jax.eval_shape(
            lambda: self.state
        )

        with (
            jax.set_mesh(self.mesh),

            self.mesh,

            nn_partitioning.axis_rules(
                get_logical_axis_rules(config)
            ),
        ):

            lowered = self.train_step.lower(
                abstract_train_state,
                shaped_batch,
            )

            self.compiled_train_step = (
                lowered.compile()
            )

        print(
            "[trainer] compile complete"
        )

        # --------------------------------------------------
        # Logger
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
                param_info["embedding_params"]
            ),

            num_devices=self.num_devices,
        )

    # ========================================================
    # Train loop
    # ========================================================

    def train(
        self,
        dataloader: Iterator,
    ):

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

        print(
            f"\nTraining for "
            f"{total_steps:,} steps\n"
        )

        prefetched_loader = (
            prefetch_to_device(
                iter(dataloader),
                size=8,
            )
        )

        data_iter = iter(
            prefetched_loader
        )

        host_step = int(
            jax.device_get(
                self.state.step
            )
        )

        while host_step < total_steps:

            # ------------------------------------------
            # Build batch
            # ------------------------------------------

            micro_batches = []

            for _ in range(
                self.grad_accum
            ):

                batch = next(
                    data_iter
                )

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

            batch = np.stack(
                micro_batches
            )

            batch = jax.device_put(
                batch,
                self.input_sharding,
            )

            # ------------------------------------------
            # TRUE synchronous timing
            # ------------------------------------------

            step_start = time.perf_counter()

            self.state, metrics = (
                self.compiled_train_step(
                    self.state,
                    batch,
                )
            )

            #
            # FORCE DEVICE SYNC
            #

            metrics = jax.tree_util.tree_map(
                lambda x: x.block_until_ready(),
                metrics,
            )

            self.state.step.block_until_ready()

            step_time = (
                time.perf_counter()
                - step_start
            )

            # ------------------------------------------
            # Logging
            # ------------------------------------------

            if (
                host_step
                % cfg.runtime.log_interval
                == 0
            ):

                metrics = (
                    jax.tree_util.tree_map(
                        lambda x: float(
                            jax.device_get(x)
                        ),
                        metrics,
                    )
                )

                current_step = int(
                    jax.device_get(
                        self.state.step
                    )
                )

                tokens_seen = int(
                    jax.device_get(
                        self.state
                        .tokens_processed
                    )
                )

                lr = _scalar(
                    self.schedule(
                        current_step
                    )
                )

                self.logger.log_step(
                    step=current_step,

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

                host_step = current_step

            else:

                host_step += 1

            # ------------------------------------------
            # Checkpoint
            # ------------------------------------------

            if (
                host_step > 0
                and host_step
                % self.checkpoint_interval
                == 0
            ):

                self.checkpoints.save(
                    host_step,
                    self.state,
                )

        # --------------------------------------------------
        # Final save
        # --------------------------------------------------

        final_step = int(
            jax.device_get(
                self.state.step
            )
        )

        final_tokens = int(
            jax.device_get(
                self.state
                .tokens_processed
            )
        )

        self.checkpoints.save(
            final_step,
            self.state,
        )

        self.checkpoints.wait()

        self.logger.log_summary(
            final_step,
            final_tokens,
        )

        self.checkpoints.close()
