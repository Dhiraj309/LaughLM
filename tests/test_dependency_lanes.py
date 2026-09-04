from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_INPUT = ROOT / "requirements" / "inputs" / "legacy-cpu-py312.in"
LEGACY_CONSTRAINTS = (
    ROOT / "requirements" / "locks" / "legacy-cpu-py312.txt"
)
PYPROJECT = ROOT / "pyproject.toml"


def _constraint_lines() -> list[str]:
    return [
        line.strip()
        for line in LEGACY_CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_legacy_input_is_the_stable_developer_install():
    assert LEGACY_INPUT.read_text(encoding="utf-8").strip().endswith(
        "-e .[dev]"
    )


def test_legacy_constraints_preserve_current_exact_runtime_pins():
    constraints = set(_constraint_lines())
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    exact_pins = {
        "jax": "0.4.38",
        "jaxlib": "0.4.38",
        "flax": "0.10.2",
        "optax": "0.2.4",
        "orbax-checkpoint": "0.11.4",
    }
    for package, version in exact_pins.items():
        assert f'"{package}=={version}"' in pyproject
        assert f"{package}=={version}" in constraints


def test_legacy_constraints_do_not_enable_optional_gen2_packages():
    constraints = "\n".join(_constraint_lines()).lower()

    assert "tokamax" not in constraints
    assert "grain" not in constraints
