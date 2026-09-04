"""Contract tests for the stable and Gen-2 GitHub Actions lanes."""

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def test_ci_defines_separate_stable_and_gen2_cpu_jobs() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "stable-cpu:" in workflow
    assert "gen2-cpu:" in workflow
    assert "name: stable CPU" in workflow
    assert "name: Gen-2 CPU" in workflow


def test_stable_lane_uses_legacy_constraints() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "python -m pip install -c requirements/locks/legacy-cpu-py312.txt "
        '-e ".[dev]"'
    ) in workflow
    assert '"jax": "0.4.38"' in workflow


def test_gen2_lane_installs_modern_input_without_legacy_pyproject_deps() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m pip install -r requirements/inputs/gen2-cpu-py312.in" in workflow
    assert "python -m pip install --no-deps -e ." in workflow
    assert '"jax": "0.11.1"' in workflow
    assert '"tokamax": "0.0.13"' in workflow


def test_both_lanes_run_required_checks() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("python -m pip check") >= 2
    assert workflow.count("tests/test_backend_config.py") >= 2
    assert workflow.count("tests/test_checkpoint_metadata.py") >= 2
    assert workflow.count("tests/test_train_state.py") >= 2
    assert "tests/test_optional_capabilities.py" in workflow
