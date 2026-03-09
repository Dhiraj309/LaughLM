import yaml
from pathlib import Path
from typing import Dict, Any

from config.schema import LaughLMConfig


# ------------------------------------------------------------
# YAML utilities
# ------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return data


# ------------------------------------------------------------
# Dictionary merge
# ------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    override values take precedence.
    """

    result = base.copy()

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value

    return result


# ------------------------------------------------------------
# Main config loader
# ------------------------------------------------------------

def load_config(
    base_config: Path,
    override_config: Path | None = None
) -> LaughLMConfig:
    """
    Load and validate configuration.

    Parameters
    ----------
    base_config : Path
        Base YAML configuration

    override_config : Path | None
        Optional override YAML

    Returns
    -------
    LaughLMConfig
        Fully validated configuration object
    """

    base_dict = _load_yaml(base_config)

    if override_config is not None:
        override_dict = _load_yaml(override_config)
        merged = _deep_merge(base_dict, override_dict)
    else:
        merged = base_dict

    config = LaughLMConfig(**merged)

    return config
