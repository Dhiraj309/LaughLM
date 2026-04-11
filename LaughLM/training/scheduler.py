import optax  
from typing import Callable  
  
from LaughLM.config.schema import LaughLMConfig  
  
  
# ------------------------------------------------------------  
# Utility: compute total training steps  
# ------------------------------------------------------------  
  
def compute_total_steps(config: LaughLMConfig) -> int:  
    """  
    Derive total gradient update steps from config.  
  
    tokens_per_step = seq_len × micro_batch × data_parallel × grad_accumulation  
    total_steps     = total_tokens / tokens_per_step  
    """  
    tokens_per_step = (  
        config.runtime.seq_len  
        * config.runtime.micro_batch_per_device  
        * config.parallelism.data_parallel  
        * config.runtime.gradient_accumulation  
    )  
    return config.runtime.total_tokens // tokens_per_step  
  
  
# ------------------------------------------------------------  
# Cosine decay with warmup  
# ------------------------------------------------------------  
  
def build_cosine_scheduler(config: LaughLMConfig) -> Callable:  
    """  
    Warmup → cosine decay.  
  
    Standard schedule for fine-tuning or short pretraining runs.  
    NOT recommended for production pretraining — use WSD instead.  
    WSD allows training extension without restart; cosine does not.  
    """  
    warmup      = config.scheduler.warmup_steps  
    lr          = config.optimizer.learning_rate  
    min_ratio   = config.scheduler.min_lr_ratio  
    total_steps = compute_total_steps(config)  
  
    return optax.warmup_cosine_decay_schedule(  
        init_value=0.0,  
        peak_value=lr,  
        warmup_steps=warmup,  
        decay_steps=total_steps,  
        end_value=lr * min_ratio,  
    )  
  
  
# ------------------------------------------------------------  
# Linear (warmup → linear decay)  
# ------------------------------------------------------------  
  
def build_linear_scheduler(config: LaughLMConfig) -> Callable:  
    """  
    Linear warmup → linear decay to zero.  
    """  
    warmup      = config.scheduler.warmup_steps  
    lr          = config.optimizer.learning_rate  
    total_steps = compute_total_steps(config)  
  
    warmup_sched = optax.linear_schedule(  
        init_value=0.0,  
        end_value=lr,  
        transition_steps=warmup,  
    )  
  
    decay_sched = optax.linear_schedule(  
        init_value=lr,  
        end_value=0.0,  
        transition_steps=total_steps - warmup,  
    )  
  
    return optax.join_schedules(  
        schedules=[warmup_sched, decay_sched],  
        boundaries=[warmup],  
    )  
  
  
# ------------------------------------------------------------  
# Inverse square root (T5 / original Transformer)  
# ------------------------------------------------------------  
  
def build_rsqrt_scheduler(config: LaughLMConfig) -> Callable:  
    """  
    Warmup then inverse square root decay.  
    LR ∝ 1/√step after warmup.  
    Used in T5 and the original Transformer paper.  
    """  
    warmup = config.scheduler.warmup_steps  
    lr     = config.optimizer.learning_rate  
  
    def schedule(step: int) -> float:  
        step = max(step, 1)  
        scale = min(  
            step ** -0.5,  
            step * warmup ** -1.5,  
        )  
        return lr * scale  
  
    return schedule  
  
  
# ------------------------------------------------------------  
# WSD — Warmup → Stable → Decay  
# ------------------------------------------------------------  
  
def build_wsd_scheduler(config: LaughLMConfig) -> Callable:  
    """  
    Warmup-Stable-Decay learning rate schedule (MiniCPM, 2024).  
    Now supports BOTH warmup_steps and warmup_fraction.  
    """  
  
    lr          = config.optimizer.learning_rate  
    min_ratio   = config.scheduler.min_lr_ratio  
    total_steps = compute_total_steps(config)  
  
    # ------------------------------------------------------------  
    # 🔥 FIX: Warmup handling (steps OR fraction)  
    # ------------------------------------------------------------  
    if config.scheduler.warmup_steps is not None:  
        warmup = config.scheduler.warmup_steps  
    elif getattr(config.scheduler, "warmup_fraction", None) is not None:  
        warmup = int(total_steps * config.scheduler.warmup_fraction)  
    else:  
        raise ValueError(  
            "WSD scheduler requires either 'warmup_steps' or 'warmup_fraction'."  
        )  
  
    # ------------------------------------------------------------  
    # Stable phase  
    # ------------------------------------------------------------  
    stable_fraction = config.scheduler.stable_fraction  
    stable_steps = int(total_steps * stable_fraction)  
  
    # ------------------------------------------------------------  
    # Decay phase  
    # ------------------------------------------------------------  
    if config.scheduler.decay_steps is not None:  
        decay_steps = config.scheduler.decay_steps  
        stable_steps = total_steps - warmup - decay_steps  
    else:  
        decay_steps = total_steps - warmup - stable_steps  
  
    # ------------------------------------------------------------  
    # 🔥 Safety checks  
    # ------------------------------------------------------------  
    if warmup < 0:  
        raise ValueError(f"warmup must be >= 0, got {warmup}")  
  
    if stable_steps < 0:  
        raise ValueError(  
            f"Computed stable_steps is negative ({stable_steps}). "  
            f"Check stable_fraction or decay_steps."  
        )  
  
    if decay_steps < 0:  
        raise ValueError(  
            f"Computed decay_steps is negative ({decay_steps}). "  
            f"Reduce warmup or stable_fraction."  
        )  
  
    if warmup + stable_steps >= total_steps:  
        raise ValueError(  
            f"Invalid schedule:\n"  
            f"  warmup: {warmup}\n"  
            f"  stable: {stable_steps}\n"  
            f"  total:  {total_steps}\n"  
            f"→ No room for decay."  
        )  
  
    min_lr = lr * min_ratio  
  
    # ------------------------------------------------------------  
    # Phase 1: Warmup  
    # ------------------------------------------------------------  
    warmup_sched = optax.linear_schedule(  
        init_value=0.0,  
        end_value=lr,  
        transition_steps=max(warmup, 1),  
    )  
  
    # ------------------------------------------------------------  
    # Phase 2: Stable  
    # ------------------------------------------------------------  
    stable_sched = optax.constant_schedule(lr)  
  
    # ------------------------------------------------------------  
    # Phase 3: Decay  
    # ------------------------------------------------------------  
    decay_sched = optax.linear_schedule(  
        init_value=lr,  
        end_value=min_lr,  
        transition_steps=max(decay_steps, 1),  
    )  
  
    schedule = optax.join_schedules(  
        schedules=[warmup_sched, stable_sched, decay_sched],  
        boundaries=[warmup, warmup + stable_steps],  
    )  
  
    return schedule  
  
def build_scheduler(config: LaughLMConfig) -> Callable:  
    """  
    Build learning rate schedule from config.  
  
    Returns a callable: step (int) → learning_rate (float).  
    This callable is passed to build_optimizer() to be baked into  
    the optimizer chain via optax.scale_by_learning_rate.  
    """  
  
    sched_type = config.scheduler.type  
  
    if sched_type == "cosine":  
        return build_cosine_scheduler(config)  
  
    if sched_type == "linear":  
        return build_linear_scheduler(config)  
  
    if sched_type == "rsqrt":  
        return build_rsqrt_scheduler(config)  
  
    if sched_type == "wsd":  
        return build_wsd_scheduler(config)  
  
    raise ValueError(  
        f"Unknown scheduler type: '{sched_type}'. "  
        f"Valid options: cosine, linear, rsqrt, wsd."  
    )
