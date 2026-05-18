"""
LaughLM/training/checkpoint.py

Frontier-grade Orbax checkpoint manager.

Frontier-grade additions
────────────────────────────────────────────
1. Async checkpoint saves
2. Mesh-native sharded restore
3. No full host gathers
4. Deterministic restore semantics
5. Safe async finalization
6. Explicit restore-failure semantics
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

        options = (
            ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                create=True,
                enable_async_checkpointing=True,
                async_options=(
                    ocp.AsyncOptions()
                ),
            )
        )

        checkpointer = (
            ocp.PyTreeCheckpointer()
        )

        self.manager = (
            ocp.CheckpointManager(
                self.directory,
                checkpointer,
                options=options,
            )
        )

        print(
            "[checkpoint] directory:\n"
            f"  {self.directory}"
        )

    # ==================================================
    # Save
    # ==================================================

    def save(
        self,
        step: int,
        state,
    ):

        if step < 0:

            raise ValueError(
                f"Invalid checkpoint "
                f"step: {step}"
            )

        print(
            f"[checkpoint] saving "
            f"step {step:,}"
        )

        save_args = (
            ocp.args.StandardSave(
                state
            )
        )

        self.manager.save(
            step,
            args=save_args,
        )

    # ==================================================
    # Wait
    # ==================================================

    def wait(self):

        self.manager.wait_until_finished()

    # ==================================================
    # Latest
    # ==================================================

    def latest_step(self):

        return (
            self.manager.latest_step()
        )

    # ==================================================
    # Restore
    # ==================================================

    def restore_latest(
        self,
        target_state,
    ):

        self.wait()

        latest_step = (
            self.latest_step()
        )

        if latest_step is None:

            print(
                "[checkpoint] "
                "no checkpoint found"
            )

            return None

        print(
            f"[checkpoint] restoring "
            f"step {latest_step:,}"
        )

        try:

            restore_args = (
                ocp.args.StandardRestore(
                    item=target_state,
                )
            )

            restored_state = (
                self.manager.restore(
                    latest_step,
                    args=restore_args,
                )
            )

            print(
                "[checkpoint] restore "
                "successful"
            )

            return (
                restored_state,
                latest_step,
            )

        except Exception as e:

            print(
                "[checkpoint] "
                "RESTORE FAILED\n"
                f"{type(e).__name__}: "
                f"{e}"
            )

            traceback.print_exc()

            raise

    # ==================================================
    # Close
    # ==================================================

    def close(self):

        self.wait()

        self.manager.close()
