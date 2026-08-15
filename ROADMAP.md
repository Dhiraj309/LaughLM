# LaughLM Roadmap

**Status:** Active planning baseline
**Primary path:** LLaMA + PMAP + TPU v5e-8 single VM
**Dataset path:** Hugging Face pre-tokenized raw `.bin` shards
**Current production config:** `configs/v5e_pmap_true135m_production.yaml`

## Scope and operating rules

This roadmap covers the maintained training and export path. The following are
deprecated and are not roadmap blockers:

- `LaughLM/data/train_tokenizer.py`
- `LaughLM/data/tokenizer.py`
- `LaughLM/data/domain_sampler.py`
- `LaughLM/model/layers/*`

Training must not be executed locally in the Windows development environment.
Runtime validation, throughput measurements, profiling, and model/JAX tests are
manual TPU gates supplied by the project owner.

Only one milestone should be active at a time. Every implementation change must
include a reviewable diff and an explicit TPU test gate before the next change.

Status flags: `[ ]` not started, `[~]` implementation in progress or awaiting
validation, `[x]` fully completed.

## Active baseline

The current production path already includes:

- PMAP over the v5e-8 local devices;
- SplashAttention for training;
- fused QKV projections;
- chunked cross-entropy with rematerialized logit chunks;
- gradient accumulation through JAX scan;
- `dots_saveable` activation rematerialization;
- host-side input prefetch;
- Hugging Face shard download and cache handling;
- persistent JAX compilation cache;
- asynchronous checkpointing;
- tied embeddings;
- a 32,064-token vocabulary.

The current configuration labels the attention variant as GQA but sets
`num_kv_heads: 8` for `num_heads: 8`, which is effectively MHA. The target
configuration is real GQA with fewer KV heads, subject to Splash TPU validation.

## Milestone overview

| Status | Milestone | Outcome | Priority |
|---|---|---|---|
| [x] | M0 | Freeze the active baseline and measurement contract; TPU baseline recorded | Complete |
| [x] | M1 | Make HF `.bin` ingestion safe and observable; TPU gate passed | Complete |
| [~] | M2 | Make configuration and architecture intent authoritative; current MHA baseline passed, real GQA pending | Blocking |
| [ ] | M3 | Make token accounting and checkpoint resume durable | Blocking |
| [ ] | M4 | Establish a measured PMAP performance baseline | High |
| [ ] | M5 | Validate and optimize real GQA + SplashAttention | High |
| [ ] | M6 | Tune memory, input pipeline, and compilation behavior | High |
| [ ] | M7 | Evaluate optional fused kernels and advanced execution paths | Future |
| [ ] | M8 | Release, export, and long-run operational gate | Final |

## M0 — Baseline and measurement contract

**Goal:** Establish one reproducible TPU baseline before optimization changes.

**Status:** [x] Run manifest and bounded TPU baseline validated.

### Features

- [x] Record the exact Python, JAX, jaxlib, Flax, Optax, Orbax, Grain, and TPU
  runtime versions used on the training VM.
- [x] Record the exact git revision, YAML config, CLI overrides, HF revision, and
  shard list for every run.
- [x] Standardize metrics for loss, learning rate, tokens/sec, step time, input
  wait, device transfer, compilation time, checkpoint time, and MFU.
- [x] Use an isolated checkpoint directory for every smoke or benchmark run.
- [x] Keep the production baseline with `kernel_backend: native` and
  `data_backend: native` until alternatives prove better on TPU.

### Exit gate

- [x] One bounded TPU run produces a complete run manifest and enough metrics to
  compare later changes. No performance claim is accepted without a baseline.

## M1 — HF binary shard safety

**Goal:** Make pre-tokenized shard loading correct, fail-fast, and measurable.

**Status:** [x] Implementation complete
**Acceptance:** [x] TPU validation passed on the selected 2-train/3-validation
shard run.

### Features

- [x] Document and validate the on-disk format: flat raw `uint16` token stream,
  no header, no index sidecar, and vocabulary IDs within range.
- [x] Keep `uint16` while vocabulary size is at most 65,535. If vocabulary size
  exceeds that limit, migrate the shard format to `uint64` and document the
  device-side input cast policy.
- [x] Reject empty or shorter-than-`seq_len + 1` shards.
- [x] Remove the fallback that silently reuses all shards when a host receives no
  assigned shard.
- [x] Propagate prefetch-thread exceptions to the training iterator.
- [x] Make requested data backend behavior explicit: fallback from Grain to native
  must be either disabled or clearly recorded as a deliberate choice.
- [x] Validate global/local batch divisibility before creating the loader.
- [x] Log resolved shard paths, byte size, dtype, token count, host assignment,
  and local batch shape before model initialization.

### Exit gate

- [x] The exact production command loads the selected train and validation
  shards, produces fixed-shape batches, and fails clearly for invalid or
  incomplete shards.
- [x] The TPU log proves that each selected shard is used according to the
  single-VM process topology.

### TPU gate for changes in M1

- [x] Run the two-train/three-validation-shard command with `--max_steps 2`
  and `--fresh`.
- [ ] Report shard paths, dtype, token counts, batch shapes, process index/count,
  first-step loss, and whether any fallback was used.

## M2 — Configuration and architecture contract

**Goal:** Ensure configuration values describe the model and execution that
actually run.

**Status:** [~] Dtype and current MHA architecture validation passed on TPU.
Real GQA deferred to M5.

### Features

- [x] Make `spmd.dtype` the canonical dtype policy for parameter, compute, and
  output dtypes.
- [x] Remove or reject the unused top-level `dtype` YAML block from the active
  production config.
- [x] Reconcile legacy `parallelism` dtype fields and checkpoint metadata during
  migration.
- [x] Set `optimizations.sharding_strategy` to a PMAP-appropriate value or clearly
  mark it as unused by the PMAP trainer.
- [ ] Require true GQA when `attention_variant: gqa`: `num_kv_heads` must be less
  than `num_heads` unless an explicit MHA mode is selected.
- [x] Validate all architecture options used by the active LLaMA implementation;
  unsupported variants fail during config loading rather than silently no-op.
- [x] Keep `fused_qkv`, tied embeddings, SwiGLU, pre-norm, RoPE, and Splash as
  explicit supported production choices; label the current 8-Q/8-KV setup as
  MHA until real GQA is available.

### Exit gate

- [ ] The resolved configuration, model factory, checkpoint metadata, and
  startup logs agree on backend, mesh, dtype, architecture, attention variant,
  and vocabulary size.

### TPU gate for changes in M2

- [x] Run model initialization and one training step with the production config.
- [x] Confirm the logs show the resolved dtypes, `num_heads`, `num_kv_heads`,
  actual attention implementation, and no unsupported-option fallback.

## M3 — Long-run state and checkpoint durability

**Goal:** Make 20B-token training resumable without counter overflow or stale
checkpoint metadata.

**Status:** [~] PMAP int64 token-counter implementation complete; TPU validation
and remaining durability work pending.

### Features

- [~] Store PMAP `tokens_processed` and per-step token increments as host-side
  `int64` values. Device-side token accumulation is intentionally disabled
  because global JAX x64 breaks SplashAttention index arithmetic; TPU
  validation and the future FSDP path remain pending.
- [x] Keep the PMAP optimizer step dtype (`int32`) separate from the host token
  count dtype (`int64`). The future FSDP state migration remains tracked
  separately.
- [~] Ensure async checkpoint restore performs the same metadata compatibility
  checks as the synchronous path. The composite manager now persists metadata
  atomically and validates it before restore; TPU validation is pending.
- [~] Ensure model state, optimizer state, token count, iterator state, and metadata
  are committed in a recoverable order. Native PMAP now records a deterministic
  next-batch index; TPU save/resume validation is pending.
- [x] Preserve the existing atomic metadata write and save-completion ordering.
- [~] Verify retention behavior with `checkpoint_max_to_keep: 1`.
  Native sidecar metadata now follows Orbax-retained steps; TPU validation of
  both native and async managers is pending.

### Exit gate

- [ ] A TPU run can save, stop, restart, restore, and continue with monotonic
  step and token counts.
- [ ] A deliberately incompatible config is rejected before training resumes.

### TPU gate for changes in M3

- [ ] Run a short save/resume cycle with async checkpointing enabled.
- [ ] Compare the restored step, `tokens_processed`, optimizer state, and next
  data batch against the preemption point.
- [ ] Test one intentionally changed dtype or architecture field and confirm
  restore rejection.

## M4 — PMAP performance baseline

**Goal:** Measure the current production path before changing kernels or mesh
behavior.

**Status:** [ ] Not started

### Features

- [ ] Measure compile time separately from steady-state step time.
- [ ] Measure input wait and device-transfer time separately from model time.
- [ ] Verify gradient accumulation is compiled as one scan and does not introduce
  host-side Python work per microbatch.
- [ ] Record Splash block size, rematerialization policy, logit chunk size, batch
  geometry, and effective tokens per optimizer step.
- [ ] Verify compilation-cache reuse on a second run.

### Exit gate

- [ ] Produce a baseline report with steady-state tokens/sec, MFU, memory
  behavior, compile time, input wait percentage, and checkpoint overhead.

### TPU gate for changes in M4

- [ ] Run the same bounded workload twice: once cold-cache and once warm-cache.
- [ ] Provide the first-step compile time, steady-state tokens/sec, loss
  stability, and input/device timing breakdown.

## M5 — Real GQA and SplashAttention

**Goal:** Move from the current MHA-equivalent setup to validated GQA.

**Status:** [ ] Not started

### Features

- [ ] Select a concrete ratio, initially `num_heads=8`, `num_kv_heads=2` or `4`.
- [ ] Verify Q/K/V projection shapes and KV-head broadcasting.
- [ ] Verify SplashAttention supports the selected GQA shape on the target JAX/TPU
  stack.
- [ ] Compare GQA against the current MHA-equivalent baseline for loss, memory,
  compile time, and tokens/sec.
- [ ] Keep `attention_fallback: error` for production so an unintended XLA fallback
  cannot masquerade as a Splash benchmark.

### Exit gate

- [ ] Real GQA runs with SplashAttention, produces finite loss, and has an
  explicit performance/memory comparison against the baseline.

### TPU gate for changes in M5

- [ ] Run baseline MHA-equivalent and real-GQA configurations for the same short
  workload.
- [ ] Report attention dispatch, compile time, step time, memory outcome, loss,
  and any fallback/error.

## M6 — Memory, input, and compilation tuning

**Goal:** Optimize the measured bottleneck without changing multiple variables
at once.

**Status:** [ ] Not started

### Experiment matrix

- [ ] `splash_block_size`: 256, 512, 1024;
- [ ] `spmd.remat.policy`: `dots_saveable` versus a less or more aggressive policy;
- [ ] logit chunk size: 2048, 4096, 8192 where memory permits;
- [ ] host prefetch depth and device-transfer scheduling;
- [ ] microbatch/gradient-accumulation pairs with constant effective tokens;
- [ ] compilation cache cold versus warm;
- [ ] checkpoint interval and async checkpoint overhead.

### Exit gate

- [ ] Select changes only when they improve steady-state throughput or memory
  while preserving loss and resume behavior.
- [ ] Every accepted setting has a recorded TPU comparison.

### TPU gate for changes in M6

- [ ] Run one controlled A/B experiment per variable.
- [ ] Report tokens/sec, peak memory, compile time, input wait, loss, and
  checkpoint time.

## M7 — Optional fused kernels and advanced execution

**Goal:** Evaluate higher-risk optimizations only after the native PMAP path is
stable and measured.

**Status:** [ ] Not started

### Features

- [ ] Validate Tokamax linear CE and SwiGLU only on the target TPU stack.
- [ ] Fix and test untied LM-head layout handling before enabling fused CE.
- [ ] Compare native XLA, chunked CE, and Tokamax with identical inputs and
  parameters.
- [ ] Revisit scanned LLaMA layers only after the unscanned production path is
  stable.
- [ ] Revisit Grain only if native memmap remains the measured bottleneck.
- [ ] Develop FSDP all-gather overlap, sequence parallelism, and 3D mesh support as
  separate future tracks; do not mix them into PMAP production changes.

### Exit gate

- [ ] An optional optimization has a working fallback, explicit dispatch
  logging, and measured TPU benefit.
- [ ] Unsupported hardware or dependency combinations fail clearly or use a
  documented fallback.

## M8 — Export and release gate

**Goal:** Make the trained checkpoint operationally useful and reproducible.

**Status:** [ ] Not started

### Features

- [ ] Validate HF export with the same vocabulary size and special-token contract.
- [ ] Verify tied embeddings and real GQA configuration in exported metadata.
- [ ] Run checkpoint-to-HF parity checks on a manually selected TPU-produced
  checkpoint or approved CPU/HF validation environment outside this local TPU
  development workflow.
- [ ] Publish the exact launch, resume, export, and shard-selection commands.
- [ ] Archive the final config, dependency versions, git revision, HF revision, and
  benchmark report.

### Exit gate

- [ ] A fresh run, resume run, export, and documented operational handoff are
  all complete, with no unresolved blocking items in M1–M7.

## Deferred tracks

These remain valid future work but are not prerequisites for the current PMAP
135M run:

- live terminal dashboard and logger v2;
- multi-host FSDP prefetch and all-gather overlap;
- 3D tensor/sequence parallelism;
- custom Pallas/Tokamax RMSNorm, SwiGLU, and cross-entropy kernels;
- broad model-architecture variants;
- legacy tokenizer, domain-sampler, and `model/layers` maintenance.

## Change protocol

Each change should include:

1. one focused implementation diff;
2. static review and `git diff --check`;
3. the exact TPU command and expected observations;
4. the user-supplied TPU log/result;
5. a decision to keep, revert, or revise the change.
