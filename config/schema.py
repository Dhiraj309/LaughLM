"""
LaughLM configuration schema.

Defines all configuration dataclasses used throughout the system.
All YAML configs are validated against these structures.

Design rules:
- YAML defines primitive values only
- Derived values computed in __post_init__
- Validation happens immediately at load time
- Config objects are immutable after creation
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# MODEL CONFIG
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_sequence_length: int

    ffn_multiplier: float = 4.0

    # Derived
    head_dim: int = field(init=False)
    ffn_hidden_size: int = field(init=False)

    def __post_init__(self):

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"GQA requires n_heads % n_kv_heads == 0 "
                f"but got {self.n_heads} % {self.n_kv_heads}"
            )

        # Head dimension
        self.head_dim = self.d_model // self.n_heads

        # FFN hidden size
        raw = int(self.d_model * self.ffn_multiplier)

        # Align to TPU-friendly multiple
        alignment = 256
        self.ffn_hidden_size = ((raw + alignment - 1) // alignment) * alignment


# ============================================================
# ARCHITECTURE FEATURES
# ============================================================

@dataclass
class ArchitectureConfig:
    use_rope: bool
    use_rmsnorm: bool
    use_swiglu: bool
    use_gqa: bool

    attention_bias: bool = False
    mlp_bias: bool = False


# ============================================================
# TOKENIZER
# ============================================================

@dataclass
class TokenizerConfig:
    algorithm: str
    vocab_size: int
    character_coverage: float
    normalization: str


# ============================================================
# DATASET CONFIG
# ============================================================

@dataclass
class DatasetSource:
    name: str
    weight: float


@dataclass
class DatasetConfig:
    sequence_length: int
    sources: List[DatasetSource]
    shuffle_buffer_size: int
    prefetch_batches: int


# ============================================================
# OPTIMIZER CONFIG
# ============================================================

@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip_norm: float


# ============================================================
# SCHEDULER CONFIG
# ============================================================

@dataclass
class SchedulerConfig:
    name: str
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


# ============================================================
# TRAINING RUNTIME
# ============================================================

@dataclass
class RuntimeConfig:
    seed: int
    precision: str
    train_steps: int
    log_every_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int


# ============================================================
# HARDWARE CONFIG
# ============================================================

@dataclass
class HardwareConfig:
    accelerator: str
    devices: int
    hbm_per_device_gb: int


# ============================================================
# PARALLELISM CONFIG
# ============================================================

@dataclass
class ParallelismConfig:
    mesh_shape: List[int]

    def __post_init__(self):

        if len(self.mesh_shape) != 2:
            raise ValueError(
                "mesh_shape must have 2 dimensions (e.g. [1,8] or [2,4])"
            )


# ============================================================
# ROOT CONFIG
# ============================================================

@dataclass
class LaughLMConfig:
    model: ModelConfig
    architecture: ArchitectureConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    hardware: HardwareConfig
    parallelism: ParallelismConfig
