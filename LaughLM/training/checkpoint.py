"""
LaughLM/training/checkpoint.py

Frontier-grade Orbax checkpoint manager.

TPU / multi-host safe.
"""

from __future__ import annotations

from pathlib import Path
import traceback

import jax
import orbax.checkpoint as ocp

from orbax.checkpoint import checkpoint_utils


class CheckpointManager:

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 3,
    ):

        self.directory = (
            Path(directory)
            .expanduser()
            .resolve()
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==================================================
        # Orbax options
        # ==================================================

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,

            #
            # async saves
            #
            enable_async_checkpointing=True,

            async_options=ocp.AsyncOptions(),

            #
            # IMPORTANT
            # Multi-host safe
            #
            enable_background_delete=True,
        )

        # ==================================================
        # Standard handler
        # ==================================================

        checkpointer = ocp.Checkpointer(
            ocp.StandardCheckpointHandler()
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            checkpointer,
            options=options,
        )

        print(
            "[checkpoint] directory:\n"
            f"  {self.directory}"
        )

        print(
            "[checkpoint] jax process:\n"
            f"  process_index="
            f"{jax.process_index()}\n"
            f"  process_count="
            f"{jax.process_count()}"
        )

    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        step: int,
        state,
    ):

        if step < 0:

            raise ValueError(
                f"Invalid checkpoint step: {step}"
            )

        print(
            f"[checkpoint] saving "
            f"step {step:,}"
        )

        try:

            #
            # Multi-host sync barrier
            #
            jax.experimental.multihost_utils.sync_global_devices(
                f"checkpoint-save-{step}"
            )

            self.manager.save(
                step,

                args=ocp.args.StandardSave(
                    state
                ),
            )

        except Exception as e:

            print(
                "[checkpoint] SAVE FAILED\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            raise

    # ======================================================
    # Wait
    # ======================================================

    def wait(self):

        self.manager.wait_until_finished()

    # ======================================================
    # Latest step
    # ======================================================

    def latest_step(self):

        return self.manager.latest_step()

    # ======================================================
    # Restore latest
    # ======================================================

    def restore_latest(
        self,
        target_state,
    ):

        #
        # ensure pending async saves finished
        #

        self.wait()

        latest_step = self.latest_step()

        if latest_step is None:

            print(
                "[checkpoint] no checkpoint found"
            )

            return None

        print(
            f"[checkpoint] restoring "
            f"step {latest_step:,}"
        )

        try:

            #
            # IMPORTANT
            #
            # Build restore args directly from
            # target sharded arrays.
            #
            # This preserves:
            # - NamedSharding
            # - GSPMD layouts
            # - mesh placement
            # - avoids host replication
            #

            restore_args = (
                checkpoint_utils
                .construct_restore_args(
                    target_state
                )
            )

            restored_state = (
                self.manager.restore(
                    latest_step,

                    args=ocp.args.StandardRestore(
                        item=target_state,
                        restore_args=restore_args,
                    ),
                )
            )

            #
            # Multi-host sync barrier
            #

            jax.experimental.multihost_utils.sync_global_devices(
                f"checkpoint-restore-{latest_step}"
            )

            print(
                "[checkpoint] restore successful"
            )

            return (
                restored_state,
                latest_step,
            )

        except Exception as e:

            print(
                "[checkpoint] RESTORE FAILED\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            raise

    # ======================================================
    # Close
    # ======================================================

    def close(self):

        try:

            self.wait()

            jax.experimental.multihost_utils.sync_global_devices(
                "checkpoint-close"
            )

        finally:

            self.manager.close()
