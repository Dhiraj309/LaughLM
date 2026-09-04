import pytest

from LaughLM.utils.checkpoint_factory import OrbaxCompositeCheckpointManager


class _FakeOrbaxManager:
    def __init__(self, saved_steps):
        self._saved_steps = list(saved_steps)

    def all_steps(self, read=True):
        assert read is True
        return list(self._saved_steps)


def _manager(tmp_path, *, saved_steps, pending_steps):
    manager = OrbaxCompositeCheckpointManager.__new__(
        OrbaxCompositeCheckpointManager
    )
    manager.directory = tmp_path
    manager.metadata_dir = tmp_path / "checkpoint_metadata"
    manager.metadata_dir.mkdir(parents=True)
    manager.manager = _FakeOrbaxManager(saved_steps)
    manager._pending_metadata = {
        int(step): {"format": "laughlm_checkpoint_v3", "step": int(step)}
        for step in pending_steps
    }
    return manager


def test_async_metadata_flush_ignores_retention_pruned_steps(tmp_path):
    manager = _manager(
        tmp_path,
        saved_steps={1000, 1500},
        pending_steps={500, 1000, 1500},
    )

    manager._flush_completed_metadata()

    assert manager._pending_metadata == {}
    assert not manager._metadata_path(500).exists()
    assert manager._metadata_path(1000).exists()
    assert manager._metadata_path(1500).exists()


def test_async_metadata_flush_rejects_missing_newest_checkpoint(tmp_path):
    manager = _manager(
        tmp_path,
        saved_steps={500},
        pending_steps={500, 1000},
    )

    with pytest.raises(RuntimeError, match="newest checkpoint.*1000"):
        manager._flush_completed_metadata()
