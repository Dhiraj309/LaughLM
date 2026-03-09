from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ------------------------------------------------------------
# Model Core Dimensions
# ------------------------------------------------------------

class ModelBaseConfig(BaseModel):
    """
    Core architectural dimensions.
    """

    d_model: int = Field(..., description="Hidden dimension size")
    num_layers: int = Field(..., description="Number of transformer blocks")
    num_heads: int = Field(..., description="Number of query attention heads")
    num_kv_heads: Optional[int] = Field(
        default=None,
        description=(
            "Number of KV heads for GQA. "
            "None = same as num_heads (standard MHA). "
            "Set to num_heads // 4 for 4:1 GQA ratio."
        )
    )

    vocab_size: int = Field(..., description="Tokenizer vocabulary size")
    max_seq_len: int = Field(..., description="Maximum sequence length")


# ------------------------------------------------------------
# Architecture Axis Selection
# ------------------------------------------------------------

class ArchitectureConfig(BaseModel):
    """
    Selects implementation along each architectural axis.
    No logic should exist in code — everything chosen here.
    """

    positional: Literal[
        "learned",
        "sinusoidal",
        "alibi",
        "rope",
        "rope_scaled"
    ]

    normalization: Literal[
        "layer_norm",
        "rms_norm",
        "deep_norm"
    ]

    norm_placement: Literal[
        "post",
        "pre",
        "sandwich"
    ]

    attention_variant: Literal[
        "mha",
        "mqa",
        "gqa",
        "mla"
    ]

    attention_impl: Literal[
        "standard",
        "flash",
        "memory_efficient"
    ]

    ffn_type: Literal[
        "gelu_mlp",
        "geglu",
        "swiglu",
        "moe"
    ]

    residual: Literal[
        "standard",
        "scaled",
        "deep_norm"
    ]

    embeddings: Literal[
        "standard",
        "scaled",
        "tied"
    ]

    bias: bool
    weight_tying: bool


# ------------------------------------------------------------
# Initialization Strategy
# ------------------------------------------------------------

class InitializationConfig(BaseModel):
    """
    Parameter initialization strategy.
    """

    method: Literal["normal", "xavier", "kaiming"]

    std: float

    embedding_std: float
    attention_std: float
    mlp_std: float

    residual_scale: float


# ------------------------------------------------------------
# Optimizer Config
# ------------------------------------------------------------

class OptimizerConfig(BaseModel):
    """
    Optimizer hyperparameters.
    """

    type: Literal["adamw", "adafactor", "lion", "muon"]

    learning_rate: float
    beta1: float
    beta2: float
    eps: float

    weight_decay: float
    gradient_clip: float


# ------------------------------------------------------------
# Scheduler Config
# ------------------------------------------------------------

class SchedulerConfig(BaseModel):
    """
    Learning rate scheduler.
    """

    type: Literal[
        "cosine",
        "linear",
        "rsqrt",
        "wsd"
    ]

    warmup_steps: int
    min_lr_ratio: float = 0.0

    # WSD-specific fields
    stable_fraction: float = Field(
        default=0.88,
        description=(
            "WSD only. Fraction of total steps in stable (constant-LR) phase. "
            "Warmup uses warmup_steps. Decay uses the remainder. "
            "0.88 → 88% stable, ~10% decay (with 2% warmup). "
            "Research finding: longer stable + short decay beats long decay."
        )
    )
    decay_steps: Optional[int] = Field(
        default=None,
        description=(
            "WSD only. Override computed decay window with a fixed step count. "
            "When None, decay_steps = total_steps - warmup_steps - stable_steps."
        )
    )


# ------------------------------------------------------------
# Runtime Training Config
# ------------------------------------------------------------

class RuntimeConfig(BaseModel):
    """
    Runtime training parameters.
    """

    seq_len: int

    micro_batch_per_device: int
    gradient_accumulation: int

    total_tokens: int

    eval_interval: int
    log_interval: int

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    checkpoint_interval: int = Field(
        default=1000,
        description="Save checkpoint every N optimizer steps"
    )

    checkpoint_max_to_keep: int = Field(
        default=3,
        description="Maximum number of recent checkpoints to retain"
    )

    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory where checkpoints are stored"
    )


# ------------------------------------------------------------
# Dataset Sources
# ------------------------------------------------------------

class DatasetSource(BaseModel):
    """
    Individual dataset source.
    """

    name: str
    weight: float
    config: Optional[str] = None
    split: str = "train"


class DataConfig(BaseModel):
    """
    Dataset pipeline configuration.
    """

    sources: List[DatasetSource]

    max_seq_len: int
    packing: bool
    eos_between_docs: bool
    pad_to_multiple: int


# ------------------------------------------------------------
# Tokenizer Config
# ------------------------------------------------------------

class TokenizerConfig(BaseModel):

    algorithm: Literal["bpe", "unigram"]
    vocab_size: int

    pre_tokenizer: Literal["byte_level"]

    number_tokenization: Literal[
        "single_digit",
        "whole_number"
    ]

    output_format: Literal[
        "huggingface_fast"
    ]


# ------------------------------------------------------------
# Parallelism Config
# ------------------------------------------------------------

class ParallelismConfig(BaseModel):

    data_parallel: int
    model_parallel: int

    compute_dtype: Literal["bfloat16", "float16"]
    param_dtype: Literal["float32", "bfloat16"]


# ------------------------------------------------------------
# Hardware Config
# ------------------------------------------------------------

class HardwareConfig(BaseModel):

    accelerator: Literal["tpu", "gpu"]
    type: str


# ------------------------------------------------------------
# Monitoring Config
# ------------------------------------------------------------

class MonitoringConfig(BaseModel):

    tensorboard: bool
    rich_terminal: bool


# ------------------------------------------------------------
# Root Config Object
# ------------------------------------------------------------

class LaughLMConfig(BaseModel):
    """
    Full experiment configuration.
    """

    model: ModelBaseConfig
    architecture: ArchitectureConfig
    initialization: InitializationConfig

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig

    data: DataConfig
    tokenizer: TokenizerConfig

    hardware: HardwareConfig
    parallelism: ParallelismConfig

    monitoring: MonitoringConfig