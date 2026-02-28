import yaml
from pathlib import Path
from typing import Dict, Any

from config.schema import (
        RootConfig,
        Modelconfig,
        AttentionConfig,
        MLPConfig,
        NormConfig,
        InitailizationConfig,
        TrainingConfig,
        OptimizerConfig,
        SchedulerConfig,
        RuntimeConfig,
        DataConfig,
        DatasetConfig,
        TokenizerConfig,
        SystemConfig,
        HardwareConfig,
        ParallelismConfig,
        ExperimentConfig,
    )


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override into base.
    Override keys must already exist in base.
    """

    result = dict(base)

    for key, value in override.items():
        if key not in result:
            raise KeyError(f"Unknown config key: {key}")

        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value


    return result


def _build_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    attention = AttentionConfig(**cfg["attention"])
    mlp = MLPConfig(**cfg["mlp"])
    norm = NormConfig(**cfg["norm"])
    intialization = IntializationConfig(**cfg["intialization"])

    return ModelConfig(
        vocab_size = cfg["vocan_size"],
        seq_len = cfg["seq_len"],
        d_model = cfg["d_model"],
        num_layers = cfg["num_layers"],
        attention = attention,
        mlp = mlp,
        norm = norm,
        intialization = intialization,
        tie_embeddings = cfg["tie_embeddings"],
    )

def _build_training_config(cfg: Dict[str, Any]) -> TrainingConfig:
    optimizer = OptimizerConfig(**cfg["optimizer"])
    scheduler = SchedulerConfig(**cfg["scheduler"])
    runtime = RuntimeConfig(**cfg["runtime"])

    return TrainingConfig(
        optimizer = optimizer,
        scheduler = scheduler,
        runtime = runtime,
    )

def _data_config(cfg: Dict[str, Any]) -> DataConfig:
    dataset = DatasetConfig(**cfg["dataset"])
    tokenizer = TokenizerConfig(**cfg["tokenizer"])

    return DataConfig(
        dataset = dataset,
        tokenizer = tokenizer,
    )

def _build_system_config(cfg: Dict[str, Any]) -> SystemConfig:
    hardware = HardwareConfig(**cfg["hardware"])
    parallelism = ParallelismConfig(**cfg["parallelism"])

    return SystemConfig(
        hardware = hardware,
        parallelism = parallelism,
    )

def _build_experiment_config(cfg: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(**cfg)

def load_config(config_root: Path) -> RootConfig:
    """
    Load full LaughLM config tree from directory.
    """

    # ---- Load Base YAMLs ----
    model_cfg = _load_yaml_file(config_root / "model" / "base.yaml")
    training_cfg = _load_yaml_file(config_root / "training" / "optimizer.yaml")
    scheduler_cfg = _load_yaml_file(config_root / "training" / "scheduler.yaml")
    runtime_cfg = _load_yaml_file(config_root / "training" / "runtime.yaml")
    data_cfg = _load_yaml_file(config_root / "data" / "dataset.yaml")
    tokenizer_cfg = _load_yaml_file(config_root / "data" / "tokenizer.yaml")
    hardware_cfg = _load_yaml_file(config_root / "system" / "hardware.yaml")
    parallel_cfg = _load_yaml_file(config_root / "system" / "parallelism.yaml")
    experiment_cfg = _load_yaml_file(config_root / "experiment" / "overrides.yaml")

    # ---- Compose structured dictionary ----
    full_model_cfg = model_cfg

    full_training_cfg = {
        "optimizer": training_cfg,
        "scheduler": scheduler_cfg,
        "runtime": runtime_cfg,
    }

    full_data_cfg = {
        "dataset": data_cfg,
        "tokenizer": tokenizer_cfg,
    }

    full_system_cfg = {
        "hardware": hardware_cfg,
        "parallelism": parallel_cfg,
    }

    # ---- Construct dataclasses ----
    model = _build_model_config(full_model_cfg)
    training = _build_training_config(full_training_cfg)
    data = _build_data_config(full_data_cfg)
    system = _build_system_config(full_system_cfg)
    experiment = _build_experiment_config(experiment_cfg)

    return RootConfig(
        model=model,
        training=training,
        data=data,
        system=system,
        experiment=experiment,
    )
