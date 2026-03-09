import yaml
from pathlib import Path
from typing import Dict, Any, Union

from config.schema import LaughLMConfig
from config.validation import validate_config


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

    # YAML may return None for empty files
    if data is None:
        data = {}

    return data


# ------------------------------------------------------------
# Dictionary merge
# ------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    The override dictionary always takes precedence.
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
# Path normalization
# ------------------------------------------------------------

def _normalize_path(path: Union[str, Path]) -> Path:
    """
    Ensure config paths are Path objects.
    """

    if isinstance(path, str):
        return Path(path)

    return path


# ------------------------------------------------------------
# Main config loader
# ------------------------------------------------------------

def load_config(
    base_config: Union[str, Path],
    override_config: Union[str, Path, None] = None
) -> LaughLMConfig:
    """
    Load and validate configuration.

    Parameters
    ----------
    base_config : str | Path
        Base YAML configuration

    override_config : str | Path | None
        Optional override YAML

    Returns
    -------
    LaughLMConfig
        Fully validated configuration object
    """

    base_config = _normalize_path(base_config)

    base_dict = _load_yaml(base_config)

    if override_config is not None:
        override_config = _normalize_path(override_config)

        override_dict = _load_yaml(override_config)

        merged = _deep_merge(base_dict, override_dict)

    else:
        merged = base_dict

    # Pydantic schema validation
    config = LaughLMConfig(**merged)

    # Cross-field validation rules
    validate_config(config)

    return config
