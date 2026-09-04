"""Tests for the explicit Gen-2 TPU dependency declaration."""

import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_gen2_tpu_lane_pins_requested_packages_and_wheel_source() -> None:
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)

    lane = project["tool"]["laughlm"]["dependency-lanes"]["gen2-tpu"]
    assert lane["find-links"] == [
        "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    ]
    assert lane["packages"] == [
        "jax[tpu]==0.11.1",
        "flax==0.12.9",
        "optax==0.2.8",
    ]


def test_stable_jax_pin_remains_the_default_project_dependency() -> None:
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)

    assert "jax==0.4.38" in project["project"]["dependencies"]
    assert "jax[tpu]==0.11.1" not in project["project"]["dependencies"]
