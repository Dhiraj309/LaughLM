"""
LaughLM/training/checkpoint.py

Modern Orbax checkpoint manager for LaughLM.

FIXES (2026)
────────────────────────────────────────────
1. Correct modern Orbax API usage
2. Async checkpoint saves
3. TPU-safe PyTree checkpointing
4. Mesh-native restore semantics
5. No handler mismatch crashes
6. Safe finalization
7. Compatible with TrainState pytrees
"""

from pathlib import Path
import traceback

import orbax.checkpoint as ocp


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
            enable_async_checkpointing=True,
            async_options=ocp.AsyncOptions(),
        )

        # ==================================================
        # MODERN API
        #
        # Use StandardCheckpointHandler
        # NOT PyTreeCheckpointer
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
            f"[checkpoint] saving step {step:,}"
        )

        try:

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

        # ensure pending async saves finished
        self.wait()

        latest_step = self.latest_step()

        if latest_step is None:

            print(
                "[checkpoint] no checkpoint found"
            )

            return None

        print(
            f"[checkpoint] restoring step "
            f"{latest_step:,}"
        )

        try:

            restored_state = (
                self.manager.restore(
                    latest_step,
                    args=ocp.args.StandardRestore(
                        target_state
                    ),
                )
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

        finally:

            self.manager.close()
