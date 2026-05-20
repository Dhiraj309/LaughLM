"""
LaughLM/training/checkpoint.py

Frontier-grade Orbax checkpoint manager.

TPU / multi-host safe.
"""

from __future__ import annotations

import json
import traceback

from pathlib import Path

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

        self.metadata_dir = (
            self.directory
            / "checkpoint_metadata"
        )

        self.metadata_dir.mkdir(
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
    # Metadata
    # ======================================================

    def _metadata_path(
        self,
        step: int,
    ) -> Path:

        return (
            self.metadata_dir
            / f"step_{step:08d}.json"
        )

    def save_metadata(
        self,
        *,
        step: int,
        metadata: dict,
    ):

        if jax.process_index() != 0:
            return

        path = self._metadata_path(step)

        with open(path, "w") as f:

            json.dump(
                metadata,
                f,
                indent=2,
                sort_keys=True,
            )

    def load_metadata(
        self,
        step: int,
    ):

        path = self._metadata_path(step)

        if not path.exists():
            return None

        with open(path, "r") as f:
            return json.load(f)

    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        step: int,
        state,
        metadata: dict | None = None,
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

            if metadata is not None:
                self.save_metadata(
                    step=step,
                    metadata=metadata,
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

        latest = self.manager.latest_step()

        if latest is None:

            print(
                "[checkpoint] no checkpoint found",
                flush=True,
            )

            return None

        print(
            f"[checkpoint] restoring step {latest}",
            flush=True,
        )

        metadata = self.load_metadata(latest)

        if metadata is not None:

            print(
                "[checkpoint] metadata:\n"
                f"  step={metadata.get('step')}\n"
                f"  tokens_processed="
                f"{metadata.get('tokens_processed')}\n"
                f"  mesh={metadata.get('mesh')}"
            )

        try:

            restored = self.manager.restore(
                latest,
                args=ocp.args.StandardRestore(
                    target_state
                ),
            )

            print(
                f"[checkpoint] restored step {latest}",
                flush=True,
            )

            return restored, latest

        except Exception as e:

            print(
                "[checkpoint] RESTORE FAILED",
                flush=True,
            )

            print(
                type(e).__name__ + ":",
                str(e),
                flush=True,
            )

            import traceback
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
