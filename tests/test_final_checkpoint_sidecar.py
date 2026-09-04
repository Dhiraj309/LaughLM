import json

import pytest

from LaughLM.training.trainer import _validate_checkpoint_sidecar


def _write_sidecar(tmp_path, *, step, tokens):
    metadata_dir = tmp_path / "checkpoint_metadata"
    metadata_dir.mkdir(parents=True)
    path = metadata_dir / f"step_{step:08d}.json"
    path.write_text(
        json.dumps(
            {
                "format": "laughlm_checkpoint_v3",
                "step": step,
                "tokens_processed": tokens,
            }
        ),
        encoding="utf-8",
    )


def test_final_checkpoint_sidecar_accepts_exact_v3_metadata(tmp_path):
    _write_sidecar(tmp_path, step=3814, tokens=3_999_268_864)
    metadata = _validate_checkpoint_sidecar(
        checkpoint_dir=tmp_path,
        expected_step=3814,
        expected_tokens=3_999_268_864,
    )
    assert metadata["format"] == "laughlm_checkpoint_v3"


def test_final_checkpoint_sidecar_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="sidecar missing"):
        _validate_checkpoint_sidecar(
            checkpoint_dir=tmp_path,
            expected_step=3814,
            expected_tokens=3_999_268_864,
        )


def test_final_checkpoint_sidecar_rejects_wrong_token_count(tmp_path):
    _write_sidecar(tmp_path, step=3814, tokens=1)
    with pytest.raises(RuntimeError, match="tokens_processed"):
        _validate_checkpoint_sidecar(
            checkpoint_dir=tmp_path,
            expected_step=3814,
            expected_tokens=3_999_268_864,
        )
