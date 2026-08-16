"""
LaughLM/config/schema.py

Full experiment configuration for LaughLM pretraining.

Frontier-grade additions:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
â€¢ SPMDConfig  â€” device mesh, logical axis rules, sharding strategy
â€¢ RematConfig â€” activation checkpointing policy + scan-over-layers
â€¢ DTypeConfig â€” explicit param / compute / output dtype separation
â€¢ LossConfig  â€” chunked logits / sparse CE configuration
â€¢ scheduler.horizon_tokens â€” separates LR schedule horizon from
  runtime stop target for safe iterative pretraining.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Model Core Dimensions
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ModelBaseConfig(BaseModel):
    """Core architectural dimensions."""

    d_model: int = Field(
        ...,
        ge=1,
        description="Hidden dimension size",
    )

    num_layers: int = Field(
        ...,
        ge=1,
        description="Number of transformer blocks",
    )

    num_heads: int = Field(
        ...,
        ge=1,
        description="Number of query attention heads",
    )

    num_kv_heads: Optional[int] = Field(
        default=None,
        description=(
            "Number of KV heads for GQA. "
            "None = same as num_heads / standard MHA."
        ),
    )

    vocab_size: int = Field(
        ...,
        ge=1,
        description="Tokenizer vocabulary size",
    )

    max_seq_len: int = Field(
        ...,
        ge=1,
        description="Maximum sequence length",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Architecture
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ArchitectureConfig(BaseModel):
    """
    Selects implementation along each architectural axis.
    No hidden code defaults should override these choices.
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

    attention_fallback: Literal[
        "warn",
        "error",
    ] = Field(
        default="warn",
        description=(
            "Behavior when attention_impl='splash' cannot use SplashAttention. "
            "'warn' falls back to XLA SDPA. "
            "'error' raises immediately, useful for benchmarks."
        ),
    )

    splash_block_size: Literal[128, 256, 512, 1024] = Field(
        default=512,
        description=(
            "Static TPU SplashAttention block size. Values must divide the "
            "sequence length for the benchmarked causal-attention path."
        ),
    )

    fused_qkv: bool = Field(
        default=False,
        description=(
            "Use a single fused qkv_proj in LLaMA attention. "
            "HF export splits qkv_proj back into q_proj/k_proj/v_proj."
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Initialization
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class InitializationConfig(BaseModel):
    """Parameter initialization strategy."""

    method: Literal[
        "normal",
        "xavier",
        "kaiming",
    ]

    std: float
    embedding_std: float
    attention_std: float
    mlp_std: float
    residual_scale: float


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Optimizer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters."""

    type: Literal[
        "adamw",
        "adafactor",
        "lion",
        "muon",
    ]

    learning_rate: float = Field(
        ...,
        gt=0.0,
    )

    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    gradient_clip: float

    mu_dtype: Literal[
        "float32",
        "bfloat16",
    ] = Field(
        default="float32",
        description=(
            "Adam first-moment dtype. "
            "Use bfloat16 to reduce optimizer-state memory; "
            "float32 is safest for convergence."
        ),
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Scheduler
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class SchedulerConfig(BaseModel):
    """
    Learning-rate scheduler configuration.

    Important:
    ----------
    runtime.total_tokens is the current stage stop target.

    scheduler.horizon_tokens is the LR schedule horizon.

    For iterative pretraining:

        runtime.total_tokens:
            1B -> 2B -> 5B -> 20B

        scheduler.horizon_tokens:
            fixed, for example 20B or 40B

    This prevents LR curve reshaping/restarting on resume.
    """

    type: Literal[
        "cosine",
        "linear",
        "rsqrt",
        "wsd",
    ]

    horizon_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional LR schedule horizon in tokens. "
            "If unset, scheduler uses runtime.total_tokens for backward compatibility. "
            "For staged/iterative training, keep this fixed while increasing "
            "runtime.total_tokens."
        ),
    )

    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
    )

    warmup_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
    )

    min_lr_ratio: float = Field(
        default=0.0,
        ge=0.0,
    )

    stable_fraction: float = Field(
        default=0.88,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of scheduler horizon steps in stable phase. "
            "Used by WSD."
        ),
    )

    decay_steps: Optional[int] = Field(
        default=None,
        ge=1,
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Loss
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LossConfig(BaseModel):
    """
    Language-modeling loss configuration.

    chunked_logits=True computes exact sparse-label CE from hidden states
    by scanning vocab chunks. This avoids materializing [B, T, vocab].
    """

    backend: Literal["native", "tokamax_linear_ce"] = Field(
        default="native",
        description=(
            "LM loss engine. native uses LaughLM chunked/dense CE; "
            "tokamax_linear_ce attempts Tokamax TPU linear-softmax CE with a safe native fallback."
        ),
    )

    tokamax_implementation: Literal["mosaic_tpu", "xla", "chunked_xla"] = Field(
        default="mosaic_tpu",
        description=(
            "Tokamax linear-softmax CE implementation used only when backend=tokamax_linear_ce. "
            "mosaic_tpu is the TPU custom-kernel experiment; unsupported dispatch falls back to native chunked CE."
        ),
    )

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
            "For vocab=49152, good values are 2048, 4096, 8192. "
            "For vocab=32064, 4096 is usually a good TPU-friendly default."
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Runtime
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class RuntimeConfig(BaseModel):
    """
    Runtime training parameters.

    Important:
    ----------
    total_tokens is the current stage stop target.

    It should not be used as the LR schedule horizon when
    scheduler.horizon_tokens is provided.
    """

    backend: Literal[
        "pmap",
        "gspmd",
        "fsdp",
        "parallel3d",
        "moe",
    ] = Field(
        default="pmap",
        description=(
            "Training backend. "
            "'pmap' = replicated data-parallel stable path. "
            "'fsdp' = mesh-native FSDP/ZeRO-style path. "
            "'gspmd' = temporary backward-compatible alias for 'fsdp'. "
            "'parallel3d' and 'moe' are reserved for future trainers."
        ),
    )
    
    @property
    def canonical_backend(self) -> str:
        """
        Canonical backend name.
    
        Backward compatibility:
          gspmd -> fsdp
    
        Do not remove the gspmd alias until all old configs/checkpoints
        have a migration path.
        """
    
        if self.backend == "gspmd":
            return "fsdp"
    
        return self.backend
    
    @property
    def backend_is_alias(self) -> bool:
        return self.backend != self.canonical_backend

    seq_len: int = Field(
        ...,
        ge=1,
    )

    micro_batch_per_device: int = Field(
        ...,
        ge=1,
    )

    gradient_accumulation: int = Field(
        ...,
        ge=1,
    )

    prefetch_size: int = Field(
        default=8,
        ge=1,
        description=(
            "CPU-side training batch prefetch queue depth. This does not "
            "transfer data to devices."
        ),
    )

    total_tokens: int = Field(
        ...,
        ge=1,
        description=(
            "Current cumulative training stop target in tokens. "
            "For staged training this may increase from 1B -> 2B -> 5B. "
            "The LR horizon should be scheduler.horizon_tokens."
        ),
    )

    eval_interval: int = Field(
        ...,
        ge=1,
    )

    eval_batches: int = Field(
        default=4,
        ge=1,
        description="Number of held-out batches averaged at each evaluation interval.",
    )

    log_interval: int = Field(
        ...,
        ge=1,
    )

    checkpoint_interval: int = Field(
        default=1000,
        ge=1,
    )

    checkpoint_max_to_keep: int = Field(
        default=3,
        ge=1,
    )

    checkpoint_dir: str = Field(
        default="checkpoints",
    )


    benchmark_mode: bool = Field(
        default=False,
        description=(
            "When true, FSDP trainer uses extra block_until_ready calls "
            "to produce cleaner timing breakdowns for benchmarking. "
            "When false, trainer avoids benchmark-only synchronization "
            "where possible for better real training throughput."
        ),
    )

    metrics_interval: int = Field(
        default=0,
        ge=0,
        description=(
            "FSDP metrics JSONL logging interval. "
            "0 = automatic: every step in benchmark_mode, otherwise log_interval. "
            "1 = every optimizer step. "
            "N > 1 = every N optimizer steps. "
            "Console logging still uses runtime.log_interval."
        ),
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Data
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

    shard_directory: str = Field(default="fineweb_edu_100bt")
    shard_filename_prefix: str = Field(default="fineweb_edu_100bt_shard")
    hf_repo_id: str = Field(
        default="LaughTaleAI/LaughLM-Tokenized-Fine",
        description="Hugging Face dataset repository containing tokenized shards.",
    )
    hf_revision: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face dataset revision; None uses the default branch.",
    )
    train_shard_start: int = Field(default=0, ge=0)
    train_shard_count: int = Field(default=1, ge=1)
    validation_shard_start: Optional[int] = Field(default=None, ge=0)
    validation_shard_count: int = Field(default=0, ge=0)
    hf_cache_dir: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face cache directory for tokenized shard files.",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tokenizer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TokenizerConfig(BaseModel):
    algorithm: Literal[
        "bpe",
        "unigram",
    ]

    vocab_size: int

    pre_tokenizer: Literal[
        "byte_level",
    ]

    number_tokenization: Literal[
        "single_digit",
        "whole_number",
    ]

    output_format: Literal[
        "huggingface_fast",
    ]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Hardware
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class HardwareConfig(BaseModel):
    accelerator: Literal[
        "tpu",
        "gpu",
    ]

    type: str


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Monitoring
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class MonitoringConfig(BaseModel):
    tensorboard: bool
    rich_terminal: bool


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SPMD Mesh
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class MeshConfig(BaseModel):
    """
    SPMD device mesh shape.

    ICI = fast local interconnect.
    DCN = data-center/network dimension.
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
    """Activation checkpointing configuration."""

    policy: Literal[
        "nothing_saveable",
        "dots_saveable",
        "dots_with_no_batch_dims_saveable",
        "everything_saveable",
    ] = Field(
        default="dots_saveable",
    )

    granularity: Literal[
        "block",
        "layer",
        "full_model",
    ] = Field(
        default="block",
    )

    scan_layers: bool = Field(
        default=True,
    )

    prevent_cse: bool = Field(
        default=False,
    )


class DTypeConfig(BaseModel):
    """
    Explicit dtype policy for parameters, computation, and output.
    """

    param_dtype: Literal[
        "float32",
        "bfloat16",
    ] = Field(
        default="float32",
    )

    compute_dtype: Literal[
        "bfloat16",
        "float16",
        "float32",
    ] = Field(
        default="bfloat16",
    )

    output_dtype: Literal[
        "float32",
        "bfloat16",
    ] = Field(
        default="float32",
    )


class SPMDConfig(BaseModel):
    """Top-level SPMD configuration block."""

    mesh: MeshConfig = Field(
        default_factory=MeshConfig,
    )

    axis_rules: LogicalAxisRules = Field(
        default_factory=LogicalAxisRules,
    )

    remat: RematConfig = Field(
        default_factory=RematConfig,
    )

    dtype: DTypeConfig = Field(
        default_factory=DTypeConfig,
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Legacy Parallelism
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ParallelismConfig(BaseModel):
    """
    Legacy parallelism config â€” kept for backward compatibility.
    """

    data_parallel: int
    model_parallel: int

    compute_dtype: Literal[
        "bfloat16",
        "float16",
    ]

    param_dtype: Literal[
        "float32",
        "bfloat16",
    ]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Profiling
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ProfilingConfig(BaseModel):
    """
    Performance profiler configuration.
    """

    enabled: bool = Field(
        default=False,
        description="Whether to enable performance profiling.",
    )

    level: Literal["off", "summary", "detailed", "developer"] = Field(
        default="summary",
        description="Profiling level depth (off, summary, detailed, developer).",
    )

    output_dir: str = Field(
        default="profiles",
        description="Root directory for profile run artifacts.",
    )

    xprof: bool = Field(
        default=False,
        description="Whether to trigger optional XProf / JAX trace collection.",
    )

    capture_device_memory_profile: bool = Field(
        default=False,
        description=(
            "Save one JAX device-memory profile after a completed warm training step. "
            "This is independent of optional XProf trace collection."
        ),
    )

    memory_profile_step: int = Field(
        default=10,
        ge=1,
        description="Optimizer step at which to save the one-shot device-memory profile.",
    )

    layer_profiling: bool = Field(
        default=False,
        description="Whether to profile individual transformer layers.",
    )

    warmup_steps: int = Field(
        default=5,
        ge=0,
        description="Number of initial warmup steps before profiling active window.",
    )

    active_steps: int = Field(
        default=100,
        ge=1,
        description="Number of steps to profile after warmup.",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Optimizations
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class OptimizationsConfig(BaseModel):
    """
    Experimental & TPU v5e optimization blueprint flags.
    Defaults preserve baseline native paths.
    """

    compilation_cache_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory for the JAX persistent compilation cache. "
            "Use a persistent TPU-working-directory path to reuse compiled "
            "executables and autotuning results across benchmark launches."
        ),
    )

    kernel_backend: Literal[
        "native",
        "tokamax",
    ] = Field(
        default="native",
        description="Kernel fusion engine. 'native' uses standard JAX, 'tokamax' uses Tokamax fused kernels.",
    )

    data_backend: Literal[
        "native",
        "grain",
    ] = Field(
        default="native",
        description="Data loader engine. 'native' uses MemmapDataset, 'grain' uses Grain DataLoader.",
    )

    sharding_strategy: Literal[
        "pmap",
        "fsdp",
        "maxtext_3d",
    ] = Field(
        default="fsdp",
        description=(
            "Sharding & mesh layout. 'pmap' is pure replicated PMAP, "
            "'fsdp' uses standard FSDP, and 'maxtext_3d' uses MaxText 3D "
            "& Sequence Parallelism."
        ),
    )

    optimizer_mu_bf16: bool = Field(
        default=False,
        description="When True, stores Adam first moment (mu) in bfloat16 to save ~33% optimizer memory.",
    )

    async_checkpointing: bool = Field(
        default=False,
        description="When True, enables Orbax async checkpointing on background host threads.",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Root Config
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LaughLMConfig(BaseModel):
    """
    Full experiment configuration for LaughLM.
    """

    model: ModelBaseConfig
    architecture: ArchitectureConfig
    initialization: InitializationConfig

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    loss: LossConfig = Field(
        default_factory=LossConfig,
    )

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

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig,
        description="Performance profiler configuration.",
    )

    optimizations: OptimizationsConfig = Field(
        default_factory=OptimizationsConfig,
        description="TPU v5e optimization blueprint options.",
    )



