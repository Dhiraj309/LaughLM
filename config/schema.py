from dataclasses import dataclass, field
from typing import Optional, Literal
import math

@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int
    dropout: float
    use_bias: bool

    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("attention dropout must be in [0, 1]")
        
@dataclass(frozen=True)
class MLPConfig:
    d_ff: int
    activation: Literal["gelu", "swiglu"]
    dropout: float
    use_bias: bool

    def __post_init__(self):
        if self.d_ff <= 0:
            raise ValueError("d_ff must be postive")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("mlp dropout must be in [0, 1]")
        
@dataclass(frozen=True)
class NormConfig:
    norm_type: Literal["layernorm", "rmsnorm"]
    eps: float
    prenorm: bool

    def __post_init__(self):
        if self.eps <= 0:
            raise ValueError("norm eps must be positive")
        
@dataclass(frozen=True)
class InitializationConfig:
    init_std: float
    residual_scale: float

    def __post__init__(self):
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be postive")
        
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    num_layers: int
    attention: AttentionConfig
    mlp: MLPConfig
    norm: NormConfig
    initialization: InitializationConfig
    tie_embeddings: bool

    def __post__init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.d_model % self.attention.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        

@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: Literal["adam"]
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: Optional[float]

    def __post__init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.beta1 <= 0:
            raise ValueError("beta1 must be in (0, 1)")
        if self.beta2 <= 0:
            raise ValueError("beta2 must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.weight_decay <= 0:
            raise ValueError("weight_decay must be >= 0")
        
@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_type: Literal["cosine", "linear"]
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float

    def __post__init__(self):
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps < 0:
            raise ValueError("total_steps must be positive")
        if not (0.0 ,+ self.min_lr_ratio <= 1.0) < 0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        

@dataclass(frozen=True)
class RuntimeConfig:
    global_batch_size: int
    microbatch_size: int
    grad_accum_steps: int
    precision: Literal["bf16", "fp32"]
    checkpoint_interval: int
    log_interval: int

    def __post__init__(self):
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        
        expected = self.microbatch_size * self.grad_accum_steps
        if self.global_batch_size != expected:
            raise ValueError("global_batch_size must be equal microbatch * grad_accum_steps")
        
@dataclass(frozen=True)
class TraininingConfig:
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: str
    shuffle: bool
    seed: int

    def __post__init__(self):
        if not self.dataset_path:
            raise ValueError("dataset_path must not empty")
        
@dataclass(frozen=True)
class TokenizerConfig:
    tokenizer_type: Literal["bpe", "sentencepiece"]
    vocab_path: str

    def __post__init__(self):
        if not self.vocab_path:
            raise ValueError("vocab_path must not be empty")
        
@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
        

@dataclass(frozen=True)
class HardwareConfig:
    num_devices: int
    device_type: Literal["tpu", "gpu", "cpu"]
    memory_per_device_gb: float

    def __post__init__(self):
        if self.num_devices <= 0:
            raise ValueError("num_devices must be positive")
        if self.memory_per_device_gb <= 0:
            raise ValueError("memory_per_device_gb must be positive")
        

@dataclass(frozen=True)
class ParallelismConfig:
    strategy: Literal["data_parallel", "model_parallel"]
    use_pmap: bool
    use_pjit: bool

    def __post__init__(self):
        if self.use_pmap and self.use_pjit:
            raise ValueError("Cannot enable both pmap and pjit")
        

@dataclass(frozen=True)
class SystemConfig:
    hardware: HardwareConfig
    parellelism: ParallelismConfig


@dataclass(frozen=True)
class ExpermentConfig:
    name: str
    seed: int

@dataclass(frozen=True)
class RootConfig:
    model: ModelConfig
    training: TraininingConfig
    data: DataConfig
    SystemConfig: SystemConfig
    experiment: ExpermentConfig