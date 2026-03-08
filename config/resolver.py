"""
LaughLM configuration resolver.

Resolves experiment configuration into the set of YAML files
needed to build the final LaughLMConfig.

Experiment YAML acts as the entrypoint describing which config
files should be used for each subsystem.
"""

from pathlib import Path
import yaml


def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_experiment(experiment_path: Path) -> dict:
    """
    Resolve experiment YAML into config file paths.

    Example experiment YAML:

    model: configs/model/base.yaml
    architecture: configs/model/architecture.yaml
    tokenizer: configs/data/tokenizer.yaml
    dataset: configs/data/dataset.yaml
    optimizer: configs/training/optimizer.yaml
    scheduler: configs/training/scheduler.yaml
    runtime: configs/training/runtime.yaml
    hardware: configs/system/hardware.yaml
    parallelism: configs/system/parallelism.yaml
    """

    experiment = _load_yaml(experiment_path)

    required_keys = [
        "model",
        "architecture",
        "tokenizer",
        "dataset",
        "optimizer",
        "scheduler",
        "runtime",
        "hardware",
        "parallelism",
    ]

    for key in required_keys:
        if key not in experiment:
            raise ValueError(
                f"Experiment config missing required section: '{key}'"
            )

    resolved_paths = {}

    for key, value in experiment.items():
        resolved_paths[key] = Path(value)

    return resolved_paths
