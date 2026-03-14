
import orbax.checkpoint as ocp
from pathlib import Path


class CheckpointManager:
    """
    Orbax checkpoint manager for LaughLM.

    Saves and restores the full training state.
    """

    def __init__(self, directory: str, max_to_keep: int = 3):

        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state",),
            options=options,
        )

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------

    def save(self, step: int, state):

        print(f"[checkpoint] saving step {step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state)
        )

        self.manager.save(step, args=args)

        print(f"[checkpoint] saved step {step}")

    # ------------------------------------------------------------
    # Restore latest checkpoint
    # ------------------------------------------------------------

    def restore_latest(self, target_state=None):

        latest_step = self.manager.latest_step()

        if latest_step is None:
            return None

        print(f"[checkpoint] restoring step {latest_step}")

        # IMPORTANT: provide target tree
        args = ocp.args.Composite(
            state=ocp.args.StandardRestore(item=target_state)
        )

        restored = self.manager.restore(
            latest_step,
            args=args,
        )

        return restored["state"], latest_step
