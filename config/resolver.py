from pathlib import Path
from typing import Dict, Any
import yaml

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_merge_strict(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override into base.
    Override keys MUST exist in base.
    """
    result = dict(base)

    for key, value in override.items():
        if key not in result:
            raise KeyError(f"Override key not found in base config: {key}")

        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _deep_merge_strict(result[key], value)
        else:
            result[key] = value

    return result


def resolve_config_tree(config_root: Path) -> Dict[str, Any]:
    """
    Returns fully merged raw config dictionary.
    """

    model_base = _load_yaml(config_root / "model" / "base.yaml")
    training_optimizer = _load_yaml(config_root / "training" / "optimizer.yaml")
    training_scheduler = _load_yaml(config_root / "training" / "scheduler.yaml")
    training_runtime = _load_yaml(config_root / "training" / "runtime.yaml")
    dataset_cfg = _load_yaml(config_root / "data" / "dataset.yaml")
    tokenizer_cfg = _load_yaml(config_root / "data" / "tokenizer.yaml")
    hardware_cfg = _load_yaml(config_root / "system" / "hardware.yaml")
    parallelism_cfg = _load_yaml(config_root / "system" / "parallelism.yaml")
    experiment_cfg = _load_yaml(config_root / "experiment" / "overrides.yaml")

    merged = {
        "model": model_base,
        "training": {
            "optimizer": training_optimizer,
            "scheduler": training_scheduler,
            "runtime": training_runtime,
        },
        "data": {
            "dataset": dataset_cfg,
            "tokenizer": tokenizer_cfg,
        },
        "system": {
            "hardware": hardware_cfg,
            "parallelism": parallelism_cfg,
        },
        "experiment": experiment_cfg,
    }

    if experiment_cfg.get("apply_overrides", False):
        overrides = experiment_cfg.get("overrides", {})
        merged = _deep_merge_strict(merged, overrides)

    return merged
