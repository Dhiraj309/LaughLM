# LaughLM — Frontier-Grade JAX/Flax LLM Training Framework

> Production-grade, TPU-first language model training framework with frontier engineering quality.

**Source**: Audited and optimized from [Dhiraj309/LaughLM PR #11](https://github.com/Dhiraj309/LaughLM/pull/11) (`perf/frontier-optim` branch)

---

## Architecture

```
LaughLM/
├── config/          # Pydantic schema, YAML loading, cross-field validation
├── data/            # Domain sampling, memmap loading, tokenizer, shard writing
├── model/
│   ├── gpt.py              # Top-level GPT with nn.scan training + for-loop inference
│   ├── transformer_block.py # Serial/parallel block with remat
│   └── layers/
│       ├── attention.py    # Splash/cuDNN/XLA dispatch + decode-specific path
│       ├── mlp.py          # GELU/GEGLU/SwiGLU with 128-aligned FFN dims
│       ├── normalization.py # RMSNorm (fp32 variance) + LayerNorm
│       ├── positional.py   # RoPE (float64 tables) + learned + sinusoidal
│       └── residual.py     # Standard/scaled/DeepNorm
├── training/
│   ├── train_step.py   # pmap step: fp32 accum → pmean → clip → optimize
│   ├── trainer.py      # Resume-safe loop with prefetch + checkpointing
│   ├── optimizer.py    # AdamW/Adafactor/Lion with masked weight decay
│   ├── scheduler.py    # Cosine/Linear/RSqrt/WSD schedules
│   ├── loss.py         # Cross-entropy + Z-loss (PaLM)
│   ├── checkpoint.py   # Async Orbax checkpointing
│   ├── logger.py       # MFU tracking + ETA + formatted output
│   └── train_state.py  # Flax dataclass with step/tokens/RNG
└── utils/
    ├── dtype.py        # Config-aware dtype resolution (no silent fallbacks)
    ├── prefetch.py     # CPU-only async buffering (no premature device transfer)
    ├── profiler.py     # TPU/GPU trace + memory + compile time measurement
    └── rng.py          # Centralized PRNG management
```

---

## Frontier Audit Findings (Corrected Assessment)

| # | Area | Severity | Fix Applied |
|---|------|----------|-------------|
| 1 | Gradient clipping after pmean | HIGH | ✅ Correct optimization semantics (not divergence) |
| 2 | FP32 gradient accumulation | HIGH | ✅ Mandatory for bf16 training |
| 3 | Attention scale vs RoPE | LOW | ✅ Convention only (mathematically equivalent) |
| 4 | Prefetch device placement | CRITICAL | ✅ CPU-only buffering |
| 5 | Missing `add_eos()` | CRITICAL | ✅ Hard runtime crash fixed |
| 6 | FFN `multiple_of` mismatch | HIGH | ✅ 128 consistently |
| 7 | Logger warmup suppression | LOW | ✅ Log from step 0 |
| 8 | NumPy-first batch handling | MEDIUM | ✅ Proper host/device staging |
| 9 | Decode-specific attention | CRITICAL | ✅ No Splash/Flash for T≤4 |
| 10 | Profiler improvements | MEDIUM | ✅ Metadata + memory + compile |

### Key Correctness Note on Gradient Clipping

The original code clipped gradients **per-device before `pmean`**. This does NOT cause replica divergence (devices still sync via pmean), but it **changes optimization semantics**:

```
clip(g_i) ≠ clip(mean(g_i))
```

Clipping local gradients biases updates toward smaller local norms. All frontier systems (MaxText, Pax, Megatron, DeepSpeed) perform: `pmean → global_norm → clip`.

### Key Note on RoPE + Scaling

RoPE is an orthonormal rotation: `R(ax) = aR(x)`. Therefore scaling before or after RoPE is **mathematically equivalent**. We scale after RoPE for convention consistency with LLaMA/MaxText, but this is NOT a correctness bug.

---

## Decode-Specific Attention (NEW)

Using Splash/Flash attention for single-token decode is pathological:
- Block sizes (128-512) designed for long sequences waste >99% compute on T=1
- Padding overhead dominates
- No O(T²) memory benefit when T=1

The fix: `causal_attention()` now routes T≤4 queries to `_decode_attention()` which uses simple XLA `dot_product_attention` with `is_causal=False` (KV cache already filters valid positions).

---

## Future Frontier Work (Priority Order)

| Priority | Task | Why |
|----------|------|-----|
| P0 | Async metrics writer | Prevent train-loop stalls |
| P0 | Eliminate host syncs in hot path | Critical for throughput |
| P1 | Sequence packing | Huge MFU gains (eliminate padding waste) |
| P1 | Static shape enforcement | Prevent recompilation storms |
| P1 | pjit migration scaffolding | Current arch is pmap-centric |
| P1 | Parameter sharding abstractions | Required before pjit |
| P2 | Optimizer state sharding | FSDP-style scaling |
| P2 | Activation sharding | Required for large models |
| P2 | Tensor parallelism | Scale beyond single-device |
| P3 | Pipeline parallelism | Multi-host scaling |
| P3 | Multi-host mesh runtime | Production multi-node |

---

## Hardware Support

| Accelerator | Training Kernel | Decode Kernel |
|-------------|----------------|---------------|
| TPU v4/v5e/v5p | Splash Attention | XLA dot_product |
| GPU (Ampere+) | cuDNN Flash | XLA dot_product |
| GPU (pre-Ampere) | XLA fallback | XLA dot_product |
| CPU | XLA fallback | XLA dot_product |

---

## Configuration

Uses Pydantic-validated YAML configs with:
- `SPMDConfig` — device mesh, logical axis rules, sharding strategy
- `RematConfig` — activation checkpointing policy + scan-over-layers
- `DTypeConfig` — explicit param/compute/output dtype separation
- `SchedulerConfig` — Cosine/Linear/RSqrt/WSD with warmup

---

## References

- [MaxText](https://github.com/AI-Hypercomputer/maxtext) — JAX/Flax reference for TPU training
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — GPU distributed training patterns
- [PaLM](https://arxiv.org/abs/2204.02311) — Z-loss, parallel blocks
- [LLaMA](https://arxiv.org/abs/2302.13971) — RMSNorm, SwiGLU, RoPE
- [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) — Parallel attention+MLP

## License

Apache 2.0
