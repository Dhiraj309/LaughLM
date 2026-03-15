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
# TPU v5e peak FLOPs
# ------------------------------------------------------------

TPU_V5E_PEAK_FLOPS = 1.576e15


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

    train_step = create_train_step(model, optimizer)

    # --------------------------------------------------------
    # Warmup (compile)
    # --------------------------------------------------------

    print("\nCompiling train step...")

    start_compile = time.time()

    params, opt_state, metrics = train_step(
        params,
        opt_state,
        dummy_batch,
    )

    jax.block_until_ready(params)

    compile_time = time.time() - start_compile

    print(f"Compilation time: {compile_time:.2f}s")

    # --------------------------------------------------------
    # Warmup iterations
    # --------------------------------------------------------

    print(f"\nRunning {warmup} warmup steps...")

    for _ in range(warmup):
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            dummy_batch,
        )

    jax.block_until_ready(params)

    # --------------------------------------------------------
    # Benchmark loop
    # --------------------------------------------------------

    print(f"\nRunning benchmark ({steps} steps)...")

    start = time.time()

    for _ in range(steps):
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            dummy_batch,
        )

    jax.block_until_ready(params)

    end = time.time()

    total_time = end - start

    step_time = total_time / steps

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    tokens_per_step = (
        config.runtime.seq_len
        * config.runtime.micro_batch_per_device
        * config.parallelism.data_parallel
        * config.runtime.gradient_accumulation
    )

    tokens_per_sec = tokens_per_step / step_time

    # --------------------------------------------------------
    # MFU calculation
    # --------------------------------------------------------

    params_count = estimate_parameters(config)["total_params"]

    mfu = (
        6 * params_count * tokens_per_sec
        / TPU_V5E_PEAK_FLOPS
    ) * 100

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nBenchmark Results")
    print("────────────────────────")

    print(f"Steps:               {steps}")
    print(f"Step time:           {step_time:.4f} s")

    print(f"Tokens / step:       {tokens_per_step:,}")
    print(f"Tokens / sec:        {tokens_per_sec:,.0f}")

    print(f"Total parameters:    {params_count:,}")

    print(f"MFU:                 {mfu:.2f}%")

    print("────────────────────────\n")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":

    benchmark(
        config_path="configs/test.yaml",
        steps=200,
        warmup=20,
    )
