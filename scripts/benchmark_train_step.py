"""
scripts/benchmark_train_step.py

Benchmark train step throughput and MFU for a LaughLM config.

FIX (audit 2025): MFU formula now includes attention FLOPs (O(S²) term)
and uses correct per-chip peak FLOPs. Old code used 6*N only and a wrong
global TPU peak. Now uses the same formula as TrainingLogger.
"""

import time
import jax
import jax.numpy as jnp

from LaughLM.config.loader import load_config
from LaughLM.model.gpt import GPTModel
from LaughLM.training.scheduler import build_scheduler
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.train_step import create_train_step
from LaughLM.model.parameter_utils import estimate_parameters
from LaughLM.utils.rng import create_rng


# ------------------------------------------------------------
# Per-chip BF16 peak FLOPs
# ------------------------------------------------------------

_TPU_FLOPS = {
    "v5e": 197e12,
    "v5p": 459e12,
    "v4":  275e12,
    "v3":  123e12,
}

_GPU_FLOPS = {
    "a100": 312e12,
    "t4":   65e12,
    "v100": 125e12,
    "l4":   121e12,
    "l40s": 362e12,
    "h100": 989e12,
    "h200": 989e12,
}


def get_hardware_flops(config) -> float:
    """Total peak BF16 FLOPs across all configured devices."""
    accel = config.hardware.accelerator
    hw_type = config.hardware.type.lower()
    devices = config.parallelism.data_parallel

    if accel == "tpu":
        for key, flops in _TPU_FLOPS.items():
            if key in hw_type:
                return flops * devices
        # Fallback
        print(f"[WARN] Unknown TPU type '{hw_type}', using v5e")
        return _TPU_FLOPS["v5e"] * devices

    if accel == "gpu":
        if hw_type in _GPU_FLOPS:
            return _GPU_FLOPS[hw_type] * devices
        print(f"[WARN] Unknown GPU type '{hw_type}', using A100")
        return _GPU_FLOPS["a100"] * devices

    raise ValueError(f"Unknown accelerator: {accel}")


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

def benchmark(config_path: str, steps: int = 200, warmup: int = 20):

    print("\nLoading config...")
    config = load_config(config_path)

    rng = create_rng(42)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("Initializing model...")

    model = GPTModel(config=config)

    batch_size = config.runtime.micro_batch_per_device
    seq_len = config.runtime.seq_len

    dummy_batch = jnp.zeros(
        (batch_size, seq_len),
        dtype=jnp.int32,
    )

    params = model.init(rng.next_key(), dummy_batch)["params"]

    # --------------------------------------------------------
    # Build optimizer
    # --------------------------------------------------------

    schedule = build_scheduler(config)
    optimizer = build_optimizer(config, schedule)

    opt_state = optimizer.init(params)

    # --------------------------------------------------------
    # Build train step
    # --------------------------------------------------------

    train_step_fn = create_train_step(
        model, optimizer,
        grad_accum=config.runtime.gradient_accumulation,
        max_grad_norm=config.optimizer.gradient_clip,
    )

    # --------------------------------------------------------
    # Build dummy batch in pmap shape
    # --------------------------------------------------------

    from LaughLM.training.train_state import TrainState
    num_devices = jax.device_count()
    grad_accum = config.runtime.gradient_accumulation

    state = TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        tokens_processed=0,
        rng_key=rng.key,
    )

    state = jax.device_put_replicated(state, jax.devices())

    # Build batch in pmap shape: (devices, grad_accum, micro_batch, seq)
    dummy_pmap_batch = jnp.zeros(
        (num_devices, grad_accum, batch_size, seq_len),
        dtype=jnp.int32,
    )

    # --------------------------------------------------------
    # Warmup (compile)
    # --------------------------------------------------------

    print("\nCompiling train step...")

    start_compile = time.time()

    state, metrics = train_step_fn(state, dummy_pmap_batch)

    jax.block_until_ready(state.params)

    compile_time = time.time() - start_compile

    print(f"Compilation time: {compile_time:.2f}s")

    # --------------------------------------------------------
    # Warmup iterations
    # --------------------------------------------------------

    print(f"\nRunning {warmup} warmup steps...")

    for _ in range(warmup):
        state, metrics = train_step_fn(state, dummy_pmap_batch)

    jax.block_until_ready(state.params)

    # --------------------------------------------------------
    # Benchmark loop
    # --------------------------------------------------------

    print(f"\nRunning benchmark ({steps} steps)...")

    start = time.time()

    for _ in range(steps):
        state, metrics = train_step_fn(state, dummy_pmap_batch)

    jax.block_until_ready(state.params)

    end = time.time()

    total_time = end - start
    step_time = total_time / steps

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    tokens_per_step = (
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * num_devices
        * grad_accum
    )

    tokens_per_sec = tokens_per_step / step_time

    # --------------------------------------------------------
    # MFU calculation (corrected: includes attention FLOPs)
    # --------------------------------------------------------

    param_info = estimate_parameters(config)
    non_emb_params = param_info["total_params"] - param_info["embedding_params"]

    n_layers = config.model.num_layers
    d_model = config.model.d_model
    seq_len_actual = seq_len - 1  # after shift_tokens

    # Standard formula (Kaplan et al. / PaLM):
    #   param_flops = 6 * N_non_emb * tokens_per_step  (fwd+bwd through weight matmuls)
    #   attn_flops  = 12 * L * d * S * tokens_per_step (O(S²) attention ops)
    param_flops = 6 * non_emb_params * tokens_per_step
    attn_flops = 12 * n_layers * d_model * seq_len_actual * tokens_per_step

    flops_per_step = param_flops + attn_flops
    flops_per_sec = flops_per_step / step_time

    hw_flops = get_hardware_flops(config)
    mfu = max(0.0, min((flops_per_sec / hw_flops) * 100, 100.0))

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(f"Model:              {param_info['total_params']:,} params ({non_emb_params:,} non-emb)")
    print(f"Hardware:           {config.hardware.type} × {num_devices} devices")
    print(f"HW peak:            {hw_flops / 1e12:.1f} TFLOPs (bf16)")
    print(f"Steps:              {steps}")
    print(f"Step time:          {step_time:.4f} s")
    print(f"Tokens / step:      {tokens_per_step:,}")
    print(f"Tokens / sec:       {tokens_per_sec:,.0f}")
    print(f"Param FLOPs/step:   {param_flops / 1e9:.2f} GFLOPS")
    print(f"Attn FLOPs/step:    {attn_flops / 1e9:.2f} GFLOPS")
    print(f"FLOPs / sec:        {flops_per_sec / 1e12:.2f} TFLOPS")
    print(f"MFU:                {mfu:.2f}%")
    print("=" * 60)
    print()


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":

    benchmark(
        config_path="configs/gpu_test.yaml",
        steps=200,
        warmup=20,
    )