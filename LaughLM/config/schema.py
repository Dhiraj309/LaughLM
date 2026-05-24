"""
LaughLM/config/schema.py

Full experiment configuration for LaughLM pretraining.

Frontier-grade additions:
──────────────────────────────────────────────────
• SPMDConfig  — device mesh, logical axis rules, sharding strategy
• RematConfig — activation checkpointing policy + scan-over-layers
• DTypeConfig — explicit param / compute / output dtype separation
• LossConfig  — chunked logits / sparse CE configuration
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ════════════════════════════════════════════════════════════════
# Model Core Dimensions
# ════════════════════════════════════════════════════════════════

class ModelBaseConfig(BaseModel):
    """Core architectural dimensions."""

    d_model: int = Field(..., description="Hidden dimension size")
    num_layers: int = Field(..., description="Number of transformer blocks")
    num_heads: int = Field(..., description="Number of query attention heads")
    num_kv_heads: Optional[int] = Field(
        default=None,
        description=(
            "Number of KV heads for GQA. "
            "None = same as num_heads (standard MHA). "
            "Set to num_heads // 4 for 4:1 GQA ratio."
        ),
    )
    vocab_size: int = Field(..., description="Tokenizer vocabulary size")
    max_seq_len: int = Field(..., description="Maximum sequence length")


# ════════════════════════════════════════════════════════════════
# Architecture Axis Selection
# ════════════════════════════════════════════════════════════════

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
        "rope_scaled",
    ]

    normalization: Literal[
        "layer_norm",
        "rms_norm",
        "deep_norm",
    ]

    norm_placement: Literal[
        "post",
        "pre",
        "sandwich",
    ]

    attention_variant: Literal[
        "mha",
        "mqa",
        "gqa",
        "mla",
    ]

    attention_impl: Literal[
        "standard",
        "xla",
        "flash",
        "cudnn",
        "memory_efficient",
        "splash",
    ]

    fused_qkv: bool = Field(
        default=False,
        description=(
            "Use a single fused qkv_proj in LLaMA attention. "
            "HF export splits qkv_proj back into q_proj/k_proj/v_proj."
        ),
    )

    attention_fallback: Literal["warn", "error"] = Field(
        default="warn",
        description=(
            "Behavior when attention_impl='splash' cannot use SplashAttention. "
            "'warn' falls back to XLA SDPA. "
            "'error' raises immediately, useful for benchmarks."
        ),
    )

    ffn_type: Literal[
        "gelu_mlp",
        "geglu",
        "swiglu",
        "moe",
    ]

    residual: Literal[
        "standard",
        "scaled",
        "deep_norm",
    ]

    embeddings: Literal[
        "standard",
        "scaled",
        "tied",
    ]

    bias: bool
    weight_tying: bool

    parallel_block: bool = Field(
        default=False,
        description=(
            "GPT-J / PaLM style parallel attention+MLP. "
            "out = x + Attn(Norm(x)) + MLP(Norm(x)) instead of serial."
        ),
    )


# ════════════════════════════════════════════════════════════════
# Initialization Strategy
# ════════════════════════════════════════════════════════════════

class InitializationConfig(BaseModel):
    """Parameter initialization strategy."""

    method: Literal["normal", "xavier", "kaiming"]
    std: float
    embedding_std: float
    attention_std: float
    mlp_std: float
    residual_scale: float


# ════════════════════════════════════════════════════════════════
# Optimizer Config
# ════════════════════════════════════════════════════════════════

class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters."""

    type: Literal["adamw", "adafactor", "lion", "muon"]
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    gradient_clip: float

    mu_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description=(
            "Adam first-moment dtype. "
            "Use bfloat16 to reduce optimizer-state memory; "
            "float32 is safest for convergence."
        ),
    )


# ════════════════════════════════════════════════════════════════
# Scheduler Config
# ════════════════════════════════════════════════════════════════

class SchedulerConfig(BaseModel):
    type: Literal["cosine", "linear", "rsqrt", "wsd"]

    warmup_steps: Optional[int] = None
    warmup_fraction: Optional[float] = None

    min_lr_ratio: float = 0.0

    stable_fraction: float = Field(
        default=0.88,
        description="Fraction of total steps in stable phase, WSD only.",
    )

    decay_steps: Optional[int] = None


# ════════════════════════════════════════════════════════════════
# Loss Config
# ════════════════════════════════════════════════════════════════

class LossConfig(BaseModel):
    """
    Language-modeling loss configuration.

    chunked_logits=True computes exact sparse-label CE from hidden states
    by scanning vocab chunks. This avoids materializing [B, T, vocab].
    """

    chunked_logits: bool = Field(
        default=False,
        description=(
            "Use exact chunked LM-head cross entropy. "
            "Avoids materializing full [batch, sequence, vocab] logits."
        ),
    )

    logits_chunk_size: int = Field(
        default=4096,
        ge=1,
        description=(
            "Vocabulary chunk size for chunked logits CE. "
            "For vocab=49152, good values are 2048, 4096, 8192."
        ),
    )

    remat_logits_chunks: bool = Field(
        default=True,
        description=(
            "Use jax.checkpoint on the vocab chunk body to reduce "
            "backward residual memory at the cost of recompute."
        ),
    )

    z_loss: float = Field(
        default=1e-4,
        ge=0.0,
        description="PaLM-style z-loss coefficient.",
    )

    ignore_index: int = Field(
        default=-100,
        description="Target id ignored by the loss.",
    )


# ════════════════════════════════════════════════════════════════
# Runtime Training Config
# ════════════════════════════════════════════════════════════════

class RuntimeConfig(BaseModel):
    """Runtime training parameters."""

    backend: Literal["pmap", "gspmd"] = Field(
        default="pmap",
        description=(
            "Training backend. "
            "'pmap' = replicated data-parallel stable path. "
            "'gspmd' = mesh-native FSDP/ZeRO-3 path."
        ),
    )

    seq_len: int
    micro_batch_per_device: int
    gradient_accumulation: int
    total_tokens: int
    eval_interval: int
    log_interval: int

    checkpoint_interval: int = Field(default=1000)
    checkpoint_max_to_keep: int = Field(default=3)
    checkpoint_dir: str = Field(default="checkpoints")


# ════════════════════════════════════════════════════════════════
# Dataset Sources
# ════════════════════════════════════════════════════════════════

class DatasetSource(BaseModel):
    """Individual dataset source."""

    name: str
    weight: float
    config: Optional[str] = None
    split: str = "train"


class DataConfig(BaseModel):
    """Dataset pipeline configuration."""

    sources: List[DatasetSource]
    max_seq_len: int
    packing: bool
    eos_between_docs: bool
    pad_to_multiple: int


# ════════════════════════════════════════════════════════════════
# Tokenizer Config
# ════════════════════════════════════════════════════════════════

class TokenizerConfig(BaseModel):
    algorithm: Literal["bpe", "unigram"]
    vocab_size: int
    pre_tokenizer: Literal["byte_level"]
    number_tokenization: Literal["single_digit", "whole_number"]
    output_format: Literal["huggingface_fast"]


# ════════════════════════════════════════════════════════════════
# Hardware Config
# ════════════════════════════════════════════════════════════════

class HardwareConfig(BaseModel):
    accelerator: Literal["tpu", "gpu"]
    type: str


# ════════════════════════════════════════════════════════════════
# Monitoring Config
# ════════════════════════════════════════════════════════════════

class MonitoringConfig(BaseModel):
    tensorboard: bool
    rich_terminal: bool


# ════════════════════════════════════════════════════════════════
# FRONTIER: SPMD Sharding Config
# ════════════════════════════════════════════════════════════════

class MeshConfig(BaseModel):
    """
    SPMD device mesh shape.

    Two-level mesh:
      ICI = fast on-chip / NVLink / TPU-ICI interconnect
      DCN = data-center network
    """

    ici_data_parallelism: int = Field(default=1, ge=1)
    ici_fsdp_parallelism: int = Field(default=1, ge=1)
    ici_tensor_parallelism: int = Field(default=1, ge=1)
    ici_sequence_parallelism: int = Field(default=1, ge=1)
    ici_pipeline_parallelism: int = Field(default=1, ge=1)

    dcn_data_parallelism: int = Field(default=1, ge=1)
    dcn_fsdp_parallelism: int = Field(default=1, ge=1)
    dcn_tensor_parallelism: int = Field(default=1, ge=1)
    dcn_pipeline_parallelism: int = Field(default=1, ge=1)

    def total_devices(self) -> int:
        return (
            (self.ici_data_parallelism * self.dcn_data_parallelism)
            * (self.ici_fsdp_parallelism * self.dcn_fsdp_parallelism)
            * (self.ici_tensor_parallelism * self.dcn_tensor_parallelism)
            * self.ici_sequence_parallelism
            * (self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism)
        )

    def axis_sizes(self) -> dict:
        return {
            "data": self.ici_data_parallelism * self.dcn_data_parallelism,
            "fsdp": self.ici_fsdp_parallelism * self.dcn_fsdp_parallelism,
            "tensor": self.ici_tensor_parallelism * self.dcn_tensor_parallelism,
            "sequence": self.ici_sequence_parallelism,
            "pipeline": self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism,
        }

    def active_axes(self) -> dict:
        return {
            k: v
            for k, v in self.axis_sizes().items()
            if v > 1
        }


class LogicalAxisRules(BaseModel):
    """
    Maps named logical tensor axes to physical mesh axis names.
    None = replicated on that axis.
    """

    batch: Optional[str] = Field(default="data")
    embed: Optional[str] = Field(default="fsdp")
    heads: Optional[str] = Field(default="tensor")
    kv_heads: Optional[str] = Field(default="tensor")
    mlp: Optional[str] = Field(default="tensor")
    vocab: Optional[str] = Field(default="fsdp")
    sequence: Optional[str] = Field(default=None)
    layers: Optional[str] = Field(default=None)


class RematConfig(BaseModel):
    """
    Activation checkpointing configuration.
    """

    policy: Literal[
        "nothing_saveable",
        "dots_saveable",
        "dots_with_no_batch_dims_saveable",
        "everything_saveable",
    ] = Field(default="dots_saveable")

    granularity: Literal[
        "block",
        "layer",
        "full_model",
    ] = Field(default="block")

    scan_layers: bool = Field(default=True)

    prevent_cse: bool = Field(default=False)


class DTypeConfig(BaseModel):
    """
    Explicit dtype policy for parameters, computation, and output.
    """

    param_dtype: Literal["float32", "bfloat16"] = Field(default="float32")

    compute_dtype: Literal[
        "bfloat16",
        "float16",
        "float32",
    ] = Field(default="bfloat16")

    output_dtype: Literal["float32", "bfloat16"] = Field(default="float32")


class SPMDConfig(BaseModel):
    """
    Top-level SPMD configuration block.
    """

    mesh: MeshConfig = Field(default_factory=MeshConfig)
    axis_rules: LogicalAxisRules = Field(default_factory=LogicalAxisRules)
    remat: RematConfig = Field(default_factory=RematConfig)
    dtype: DTypeConfig = Field(default_factory=DTypeConfig)


# ════════════════════════════════════════════════════════════════
# LEGACY: Parallelism Config
# ════════════════════════════════════════════════════════════════

class ParallelismConfig(BaseModel):
    """
    Legacy parallelism config — kept for backward compatibility.
    """

    data_parallel: int
    model_parallel: int

    compute_dtype: Literal["bfloat16", "float16"]
    param_dtype: Literal["float32", "bfloat16"]


# ════════════════════════════════════════════════════════════════
# Root Config Object
# ════════════════════════════════════════════════════════════════

class LaughLMConfig(BaseModel):
    """
    Full experiment configuration for LaughLM.
    """

    model: ModelBaseConfig
    architecture: ArchitectureConfig
    initialization: InitializationConfig

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    # IMPORTANT:
    # This field is required because trainer.py passes config.loss
    # into create_train_step/create_eval_step.
    loss: LossConfig = Field(default_factory=LossConfig)

    runtime: RuntimeConfig

    data: DataConfig
    tokenizer: TokenizerConfig

    hardware: HardwareConfig
    parallelism: ParallelismConfig

    monitoring: MonitoringConfig

    spmd: SPMDConfig = Field(
        default_factory=SPMDConfig,
        description=(
            "SPMD sharding, rematerialization, and dtype config. "
            "Defaults are safe for single-device and PMAP training."
        ),
    )
