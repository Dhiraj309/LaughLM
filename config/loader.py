
"""
LaughLM configuration loader.

Loads YAML configuration files and converts them into validated
dataclass objects defined in schema.py.
"""

from pathlib import Path
import yaml

from .schema import (
    ModelConfig,
    ArchitectureConfig,
    TokenizerConfig,
    DatasetConfig,
    DatasetSource,
    OptimizerConfig,
    SchedulerConfig,
    RuntimeConfig,
    HardwareConfig,
    ParallelismConfig,
    LaughLMConfig,
)


def _load_yaml(path: Path):
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_dataset_sources(source_list):
    """Convert dataset sources into DatasetSource objects."""
    sources = []
    for s in source_list:
        sources.append(DatasetSource(**s))
    return sources


def load_config(config_paths: dict) -> LaughLMConfig:
    """
    Load LaughLM configuration from YAML files.

    Args:
        config_paths: dictionary mapping section name -> YAML path

    Returns:
        LaughLMConfig object
    """

    model_yaml = _load_yaml(config_paths["model"])
    arch_yaml = _load_yaml(config_paths["architecture"])
    tokenizer_yaml = _load_yaml(config_paths["tokenizer"])
    dataset_yaml = _load_yaml(config_paths["dataset"])
    optimizer_yaml = _load_yaml(config_paths["optimizer"])
    scheduler_yaml = _load_yaml(config_paths["scheduler"])
    runtime_yaml = _load_yaml(config_paths["runtime"])
    hardware_yaml = _load_yaml(config_paths["hardware"])
    parallel_yaml = _load_yaml(config_paths["parallelism"])

    model = ModelConfig(**model_yaml)
    architecture = ArchitectureConfig(**arch_yaml)
    tokenizer = TokenizerConfig(**tokenizer_yaml)

    dataset_yaml["sources"] = _parse_dataset_sources(dataset_yaml["sources"])
    dataset = DatasetConfig(**dataset_yaml)

    optimizer = OptimizerConfig(**optimizer_yaml)
    scheduler = SchedulerConfig(**scheduler_yaml)
    runtime = RuntimeConfig(**runtime_yaml)
    hardware = HardwareConfig(**hardware_yaml)
    parallelism = ParallelismConfig(**parallel_yaml)

    config = LaughLMConfig(
        model=model,
        architecture=architecture,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime=runtime,
        hardware=hardware,
        parallelism=parallelism,
    )

    return config
