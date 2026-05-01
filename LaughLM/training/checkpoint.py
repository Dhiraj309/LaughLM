import orbax.checkpoint as ocp
from pathlib import Path


class CheckpointManager:
    """
    Async-capable Orbax checkpoint manager for LaughLM.

    Key change: async_options=ocp.AsyncOptions() makes manager.save()
    return immediately and write in a background thread.
    Call .wait() before restore or at end of training.
    """

    def __init__(self, directory: str, max_to_keep: int = 3):

        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            async_options=ocp.AsyncOptions(),   # ← KEY: non-blocking saves
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state",),
            options=options,
        )

    # ------------------------------------------------------------
    # Save (non-blocking — returns immediately)
    # ------------------------------------------------------------

    def save(self, step: int, state):
        print(f"[checkpoint] saving step {step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state)
        )

        # Returns immediately; write happens in background thread.
        # Do NOT print "saved" here — it isn't saved yet.
        self.manager.save(step, args=args)

    # ------------------------------------------------------------
    # Block until any in-flight save finishes
    # Call this: (a) at end of training, (b) before restore
    # ------------------------------------------------------------

    def wait(self):
        self.manager.wait_until_finished()
        print("[checkpoint] write complete")

    # ------------------------------------------------------------
    # Restore latest checkpoint
    # ------------------------------------------------------------

    def restore_latest(self, target_state=None):

        # Must drain any pending async write before reading
        self.manager.wait_until_finished()

        latest_step = self.manager.latest_step()

        if latest_step is None:
            return None

        print(f"[checkpoint] restoring step {latest_step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardRestore(item=target_state)
        )

        restored = self.manager.restore(
            latest_step,
            args=args,
        )

        return restored["state"], latest_step