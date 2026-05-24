"""
LaughLM/config/schema.py

Full experiment configuration for LaughLM pretraining.

Frontier-grade additions (perf/frontier-optim):
──────────────────────────────────────────────────
• SPMDConfig  — device mesh, logical axis rules, sharding strategy
• RematConfig — activation checkpointing policy + scan-over-layers
• DTypeConfig — explicit param / compute / output dtype separation
  (replaces the old ParallelismConfig.compute_dtype / param_dtype)

Design references:
  MaxText  (AI-Hypercomputer/maxtext) → configs/base.yml
  Levanter (stanford-crfm/levanter)   → src/levanter/trainer.py
"""

from pydantic import BaseModel, Field, model_validator
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
        "learned", "sinusoidal", "alibi",
        "rope", "rope_scaled",
    ]
    normalization: Literal["layer_norm", "rms_norm", "deep_norm"]
    norm_placement: Literal["post", "pre", "sandwich"]

    attention_variant: Literal["mha", "mqa", "gqa", "mla"]

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

    ffn_type: Literal["gelu_mlp", "geglu", "swiglu", "moe"]
    residual: Literal["standard", "scaled", "deep_norm"]
    embeddings: Literal["standard", "scaled", "tied"]

    bias: bool
    weight_tying: bool

    # ── Frontier additions ──────────────────────────────────────
    parallel_block: bool = Field(
        default=False,
        description=(
            "GPT-J / PaLM style parallel attention+MLP. "
            "out = x + Attn(Norm(x)) + MLP(Norm(x)) instead of serial. "
            "Saves one all-reduce in tensor-parallel and enables better "
            "pipelining on TPU/GPU. Used by PaLM, GPT-J, MPT."
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
        description="Fraction of total steps in stable phase (WSD only)",
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
#  FRONTIER: SPMD Sharding Config (MaxText-style)
# ════════════════════════════════════════════════════════════════

class MeshConfig(BaseModel):
    """
    SPMD device mesh shape.

    Two-level mesh following MaxText convention:
      ICI = fast on-chip / NVLink / TPU-ICI interconnect (within a host group)
      DCN = data-center network (between host groups / pod slices)

    For single-host GPU runs: all dcn_* stay at 1.
    The total device count must equal jax.device_count() at runtime.

    mesh_shape[axis] = ici_<axis> × dcn_<axis>
    Logical axis names: ["data", "fsdp", "tensor", "sequence", "pipeline"]

    Reference: AI-Hypercomputer/maxtext → configs/base.yml
    """

    # ICI (intra-host fast interconnect)
    ici_data_parallelism: int = Field(
        default=1, ge=1,
        description="Pure data-parallel replicas on fast interconnect.",
    )
    ici_fsdp_parallelism: int = Field(
        default=1, ge=1,
        description="FSDP shards on fast interconnect (params + grads sharded).",
    )
    ici_tensor_parallelism: int = Field(
        default=1, ge=1,
        description="Tensor (model) parallel degree on fast interconnect.",
    )
    ici_sequence_parallelism: int = Field(
        default=1, ge=1,
        description="Sequence/context parallel on fast interconnect.",
    )
    ici_pipeline_parallelism: int = Field(
        default=1, ge=1,
        description="Pipeline stages on fast interconnect.",
    )

    # DCN (between-host / multi-node / multi-slice)
    dcn_data_parallelism: int = Field(
        default=1, ge=1,
        description="Data-parallel replicas across hosts.",
    )
    dcn_fsdp_parallelism: int = Field(
        default=1, ge=1,
        description="FSDP shards across hosts (multi-node ZeRO-3).",
    )
    dcn_tensor_parallelism: int = Field(
        default=1, ge=1,
        description="Tensor parallel across hosts (rare — bandwidth-sensitive).",
    )
    dcn_pipeline_parallelism: int = Field(
        default=1, ge=1,
        description="Pipeline stages across hosts (inter-node).",
    )

    def total_devices(self) -> int:
        """Total device count implied by this mesh config."""
        return (
            (self.ici_data_parallelism * self.dcn_data_parallelism)
            * (self.ici_fsdp_parallelism * self.dcn_fsdp_parallelism)
            * (self.ici_tensor_parallelism * self.dcn_tensor_parallelism)
            * self.ici_sequence_parallelism
            * (self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism)
        )

    def axis_sizes(self) -> dict:
        """Collapsed logical axis sizes: {axis_name: ici * dcn}."""
        return {
            "data":     self.ici_data_parallelism     * self.dcn_data_parallelism,
            "fsdp":     self.ici_fsdp_parallelism     * self.dcn_fsdp_parallelism,
            "tensor":   self.ici_tensor_parallelism   * self.dcn_tensor_parallelism,
            "sequence": self.ici_sequence_parallelism,
            "pipeline": self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism,
        }

    def active_axes(self) -> dict:
        """Only axes with size > 1 (used for mesh construction)."""
        return {k: v for k, v in self.axis_sizes().items() if v > 1}


class LogicalAxisRules(BaseModel):
    """
    Maps named logical tensor axes to physical mesh axis names.
    None = replicated on that axis. Follows MaxText / Levanter convention.

    Consumed downstream by:
        jax.sharding.NamedSharding(mesh, PartitionSpec(...))

    The mapping tells the sharding system:
      "when you see a tensor dimension called 'embed', shard it
       across the 'fsdp' mesh axis"

    Reference: stanford-crfm/levanter → src/levanter/trainer.py
    """

    batch:    Optional[str] = Field(default="data",   description="Batch dim → mesh axis")
    embed:    Optional[str] = Field(default="fsdp",   description="Hidden/embed dim → mesh axis")
    heads:    Optional[str] = Field(default="tensor", description="Attention Q-heads → mesh axis")
    kv_heads: Optional[str] = Field(default="tensor", description="KV heads → mesh axis")
    mlp:      Optional[str] = Field(default="tensor", description="MLP intermediate → mesh axis")
    vocab:    Optional[str] = Field(default="fsdp",   description="Vocab dim → mesh axis")
    sequence: Optional[str] = Field(default=None,     description="Sequence len → mesh axis (set to 'sequence' for SP)")
    layers:   Optional[str] = Field(default=None,     description="Layer dim (scan) → mesh axis")


class RematConfig(BaseModel):
    """
    Activation checkpointing (rematerialization) config.

    Controls how much activation memory to trade for recomputation time.
    Applied per transformer block by default.

    Policy options (maps to jax.checkpoint_policies.*):
    ─────────────────────────────────────────────────────────────────
    nothing_saveable               → recompute everything (minimum memory)
    dots_saveable                  → save matmul outputs only (best balance ✓)
    dots_with_no_batch_dims_saveable → save non-batched matmuls
    everything_saveable            → save all (no remat, max memory)

    Reference: AI-Hypercomputer/maxtext → layers.py (remat_policy dispatch)
    """

    policy: Literal[
        "nothing_saveable",
        "dots_saveable",
        "dots_with_no_batch_dims_saveable",
        "everything_saveable",
    ] = Field(
        default="dots_saveable",
        description=(
            "Checkpoint policy for jax.checkpoint / nn.remat. "
            "'dots_saveable' saves matmul results, recomputes norms, "
            "activations, softmax — best memory/compute tradeoff."
        ),
    )

    granularity: Literal["block", "layer", "full_model"] = Field(
        default="block",
        description=(
            "'block' = per transformer block (standard). "
            "'layer' = per sub-layer (attention/MLP separately). "
            "'full_model' = wrap entire forward."
        ),
    )

    scan_layers: bool = Field(
        default=True,
        description=(
            "Stack transformer blocks with nn.scan. "
            "O(1) XLA compile time regardless of depth. "
            "Pairs with remat for frontier-grade memory efficiency."
        ),
    )

    prevent_cse: bool = Field(
        default=False,
        description=(
            "Pass prevent_cse=True to nn.remat if XLA CSE "
            "defeats checkpointing across loop iterations. Usually False."
        ),
    )


class DTypeConfig(BaseModel):
    """
    Explicit dtype policy for parameters, computation, and output.

    Replaces the old ParallelismConfig.compute_dtype / param_dtype
    with a more complete specification following MaxText convention.

    Standard frontier recipe:
      param_dtype   = float32  (master weights — full precision)
      compute_dtype = bfloat16 (activations + matmuls — 2× throughput)
      output_dtype  = float32  (loss accumulation — numerical stability)

    Reference: AI-Hypercomputer/maxtext → configs/base.yml (dtype, weight_dtype)
    """

    param_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description=(
            "Storage dtype for parameters and optimizer state. "
            "float32 for training stability. bfloat16 for pure-bf16 training."
        ),
    )

    compute_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description=(
            "Compute dtype for forward/backward pass — activations and matmuls. "
            "bfloat16 gives ~2× throughput on A100/H100/TPU vs float32."
        ),
    )

    output_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description=(
            "Dtype for layer outputs and loss accumulation. "
            "Always float32 to prevent loss scaling issues."
        ),
    )


class SPMDConfig(BaseModel):
    """
    Top-level SPMD configuration block.

    Groups all sharding, rematerialization, and dtype policy into one
    coherent block. Every downstream file (model, training, data) reads
    from here instead of scattered fields.

    Reference: MaxText base.yml + Levanter trainer.py
    """

    mesh: MeshConfig = Field(default_factory=MeshConfig)
    axis_rules: LogicalAxisRules = Field(default_factory=LogicalAxisRules)
    remat: RematConfig = Field(default_factory=RematConfig)
    dtype: DTypeConfig = Field(default_factory=DTypeConfig)


# ════════════════════════════════════════════════════════════════
#  LEGACY: Parallelism Config (backward-compatible wrapper)
# ════════════════════════════════════════════════════════════════

class ParallelismConfig(BaseModel):
    """
    Legacy parallelism config — kept for backward compatibility.

    New code should read from LaughLMConfig.spmd instead.
    The old data_parallel / model_parallel / compute_dtype / param_dtype
    fields are preserved so existing YAML configs don't break.
    """

    data_parallel: int
    model_parallel: int

    compute_dtype: Literal["bfloat16", "float16"]
    param_dtype: Literal["float32", "bfloat16"]


# ════════════════════════════════════════════════════════════════
#  Root Config Object
# ════════════════════════════════════════════════════════════════

class LaughLMConfig(BaseModel):
    """
    Full experiment configuration for LaughLM.

    ┌──────────────────────────────────────────────────────────┐
    │  New in perf/frontier-optim:                             │
    │                                                          │
    │  spmd: SPMDConfig       ← sharding + remat + dtype       │
    │    ├── mesh              (MaxText-style device mesh)      │
    │    ├── axis_rules        (logical → physical axis map)    │
    │    ├── remat             (activation checkpointing)       │
    │    └── dtype             (param / compute / output)       │
    │                                                          │
    │  architecture.parallel_block  ← GPT-J parallel attn+MLP │
    │                                                          │
    │  The old 'parallelism' block is kept for compat but      │
    │  new code should read from 'spmd'.                       │
    └──────────────────────────────────────────────────────────┘
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

    # ── Frontier SPMD config (new) ──────────────────────────
    spmd: SPMDConfig = Field(
        default_factory=SPMDConfig,
        description=(
            "SPMD sharding, rematerialization, and dtype config. "
            "New code should read dtypes and sharding from here. "
            "Defaults are safe for single-device training."
        ),
    )
