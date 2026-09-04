from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEN2_CPU = ROOT / "requirements" / "inputs" / "gen2-cpu-py312.in"
GEN2_TPU = ROOT / "requirements" / "inputs" / "gen2-tpu-v5e-py312.in"
README = ROOT / "requirements" / "README.md"


def _requirements(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gen2_cpu_input_has_the_candidate_modern_stack():
    requirements = _requirements(GEN2_CPU)
    expected = {
        "jax==0.11.1",
        "jaxlib==0.11.1",
        "flax==0.12.9",
        "optax==0.2.8",
        "orbax-checkpoint==0.12.4",
        "tokamax==0.0.13",
        "xprof==2.23.1",
        "grain==0.2.18",
        "pydantic==2.13.5",
    }

    assert expected <= requirements
    assert not any(line.startswith("-e ") for line in requirements)
    assert "libtpu==0.0.47" not in requirements


def test_gen2_tpu_input_adds_only_the_tpu_runtime_layer():
    requirements = _requirements(GEN2_TPU)

    assert "-r gen2-cpu-py312.in" in requirements
    assert "libtpu==0.0.47" in requirements
    assert len(requirements) == 2


def test_gen2_install_instructions_do_not_resolve_legacy_metadata():
    readme = README.read_text(encoding="utf-8")

    assert "pip install --no-deps -e ." in readme
    assert "gen2-cpu-py312.in" in readme
    assert "gen2-tpu-v5e-py312.in" in readme


def test_base_laughlm_import_has_no_gen2_package_requirement():
    import LaughLM

    assert LaughLM.__name__ == "LaughLM"
