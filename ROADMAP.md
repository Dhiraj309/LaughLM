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

The active production configuration uses explicit MHA with
`num_kv_heads: 8` and `num_heads: 8`. Real GQA requires fewer KV heads and is
tracked as an M5 TPU experiment.

## Milestone overview

| Status | Milestone | Outcome | Priority |
|---|---|---|---|
| [x] | M0 | Freeze the active baseline and measurement contract; TPU baseline recorded | Complete |
| [x] | M1 | Make HF `.bin` ingestion safe and observable; TPU gate passed | Complete |
| [~] | M2 | Make configuration and architecture intent authoritative; current MHA baseline passed, real GQA pending | Blocking |
| [~] | M3 | Make token accounting and checkpoint resume durable; PMAP save/resume gate passed | Blocking |
| [~] | M4 | Establish a measured PMAP performance baseline; timing instrumentation added | High |
| [~] | M5 | Validate and optimize real GQA + SplashAttention | High |
| [~] | M6 | Tune memory, input pipeline, and compilation behavior | High |
| [~] | M7 | Evaluate optional fused kernels and advanced execution paths | Future |
| [~] | M8 | Release, export, and long-run operational gate | Final |

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
- [~] Report shard paths, dtype, token counts, batch shapes, process index/count,
  first-step loss, and whether any fallback was used. Run manifests now persist
  shard paths, byte sizes, inferred dtype, alignment, and token counts, and the
  static run audit now verifies that shard contract; batch, process, loss, and
  fallback evidence still requires the TPU log.

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
  mark it as unused by the PMAP trainer. The active PMAP launcher now warns when
  a non-PMAP strategy such as `fsdp` is present.
- [x] Require true GQA when `attention_variant: gqa`: `num_kv_heads` must be less
  than `num_heads`; equal head counts must use explicit MHA mode.
- [x] Validate all architecture options used by the active LLaMA implementation;
  unsupported variants fail during config loading rather than silently no-op.
- [x] Keep `fused_qkv`, tied embeddings, SwiGLU, pre-norm, RoPE, and Splash as
  explicit supported production choices; label the current 8-Q/8-KV setup as
  MHA until real GQA is available.

### Exit gate

- [~] The resolved configuration, model factory, checkpoint metadata, and
  startup logs now share an explicit execution contract. TPU validation and
  migration coverage for older v3 checkpoints remain pending.

### TPU gate for changes in M2

- [x] Run model initialization and one training step with the production config.
- [x] Confirm the logs show the resolved dtypes, `num_heads`, `num_kv_heads`,
  actual attention implementation, and no unsupported-option fallback.

## M3 — Long-run state and checkpoint durability

**Goal:** Make 20B-token training resumable without counter overflow or stale
checkpoint metadata.

**Status:** [~] PMAP save/resume validation passed; retention verification and
deliberate incompatible-config rejection remain pending.

### Features

- [x] Store PMAP `tokens_processed` and per-step token increments as host-side
  `int64` values. Device-side token accumulation is intentionally disabled
  because global JAX x64 breaks SplashAttention index arithmetic. The PMAP TPU
  resume gate passed; the future FSDP path remains tracked separately.
- [x] Keep the PMAP optimizer step dtype (`int32`) separate from the host token
  count dtype (`int64`). The future FSDP state migration remains tracked
  separately.
- [x] Ensure async checkpoint restore performs the same metadata compatibility
  checks as the synchronous path. The composite manager now persists metadata
  atomically and validates it before restore; the TPU resume gate passed.
- [x] Ensure model state, optimizer state, token count, iterator state, and metadata
  are committed in a recoverable order. Native PMAP restores the deterministic
  next-batch index; TPU resume restored step 50, token count, and batch index
  before continuing to step 60.
- [x] Preserve the existing atomic metadata write and save-completion ordering.
- [~] Verify retention behavior with `checkpoint_max_to_keep: 1`.
  Native sidecar metadata now follows Orbax-retained steps, including for
  state-only saves; a dependency-light artifact audit is available, but TPU
  validation of both native and async managers is pending.

### Exit gate

- [x] A TPU run can save, stop, restart, restore, and continue with monotonic
  step and token counts.
- [~] A deliberately incompatible config is rejected before training resumes.
  Model, optimizer/scheduler, layout/dtype, and execution-contract mismatches
  now fail before restore; the static compatibility preflight now reports the
  exact mismatch paths, but TPU confirmation remains pending.

### TPU gate for changes in M3

- [x] Run a short save/resume cycle with async checkpointing enabled.
- [~] Compare the restored step, `tokens_processed`, optimizer state, and next
  data batch against the preemption point. Step, token count, and deterministic
  next-batch index were confirmed in the TPU log; exact optimizer-tree equality
  still needs an explicit comparison.
- [ ] Test one intentionally changed dtype or architecture field and confirm
  restore rejection.

## M4 — PMAP performance baseline

**Goal:** Measure the current production path before changing kernels or mesh
behavior.

**Status:** [~] PMAP timing instrumentation is implemented; cold-cache and
warm-cache TPU validation remains pending.

### Features

- [~] Measure compile time separately from steady-state step time. The first
  device step now records compile-plus-execute time; TPU validation is needed to
  separate compilation from execution precisely.
- [x] Measure input wait, host batch preparation, and device-transfer time
  separately from model time in PMAP metrics.
- [~] Verify gradient accumulation is compiled as one scan and does not introduce
  host-side Python work per microbatch. The active PMAP train step uses
  `jax.lax.scan`; TPU validation remains pending.
- [x] Record Splash block size, rematerialization policy, logit chunk size, batch
  geometry, and effective tokens per optimizer step. The launcher now reports
  the resolved rematerialization, scan, logit, head, and cache settings;
  Splash reports its selected block size during model initialization.
- [~] Verify compilation-cache reuse on a second run. The run manifest now
  records the cache directory and pre-run file count, allowing cold/warm TPU
  runs to be compared; a static run-artifact audit now checks provenance,
  timing fields, metric ordering, and value ranges, while TPU confirmation
  remains pending.

### Exit gate

- [~] Produce a baseline report with steady-state tokens/sec, MFU, memory
  behavior, compile time, input wait percentage, and checkpoint overhead. The
  static report generator and checkpoint timing artifact are implemented; TPU
  evidence remains pending.

### TPU gate for changes in M4

- [ ] Run the same bounded workload twice: once cold-cache and once warm-cache.
- [ ] Provide the first-step compile time, steady-state tokens/sec, loss
  stability, and input/device timing breakdown.

## M5 — Real GQA and SplashAttention

**Goal:** Move from the current MHA-equivalent setup to validated GQA.

**Status:** [~] TPU smoke and warm comparison passed; longer stability and
peak-memory evidence remain pending.

### Features

- [x] Select the initial comparison ratio as `num_heads=8`, `num_kv_heads=4`;
  keep the production MHA baseline unchanged until the TPU comparison.
- [x] Provide an isolated GQA override config and launcher overlay support so
  the MHA baseline is not copied or overwritten.
- [x] Provide a complete isolated MHA comparison config with the validated
  2-train/3-validation-shard geometry.
- [x] Add a dependency-light config-matrix audit for attention labels and
  Q/KV head geometry before TPU allocation.
- [x] Record the requested attention implementation, fallback policy, head
  geometry, and expected Splash GQA expansion in each run manifest; actual
  dispatch still requires TPU log confirmation.
- [x] Verify Q/K/V projection shapes and KV-head broadcasting. Compact K/V
  projections and caches are preserved; the TPU smoke run confirmed the
  expected 4-to-8 Splash boundary expansion.
- [x] Verify SplashAttention supports the selected GQA shape on the target JAX/TPU
  stack. The 5-step smoke and 50-step warm comparison completed without
  fallback or attention-shape errors.
- [~] Compare GQA against the current MHA-equivalent baseline for loss, memory,
  compile time, and tokens/sec. The 50-step comparison reports about 3.8%
  higher GQA throughput and finite loss; peak-memory artifacts and a longer
  stability window remain pending.
- [x] Keep `attention_fallback: error` for production so an unintended XLA fallback
  cannot masquerade as a Splash benchmark.

### Exit gate

- [~] Real GQA runs with SplashAttention produce finite loss and an explicit
  performance comparison; peak-memory evidence and longer stability remain.

### TPU gate for changes in M5

- [x] Run baseline MHA-equivalent and real-GQA configurations for the same
  50-step workload.
- [~] Report attention dispatch, compile time, step time, memory outcome, loss,
  and any fallback/error. Dispatch, timing, loss, and fallback evidence are
  recorded; actual peak-memory capture remains.

## M6 — Memory, input, and compilation tuning

**Goal:** Optimize the measured bottleneck without changing multiple variables
at once.

**Status:** [~] Controlled tuning overlays and opt-in memory capture are
implemented; TPU validation remains pending.

### Experiment matrix

- [~] `splash_block_size`: 256, 512, 1024 overlays are isolated in the M6
  experiment matrix; TPU selection remains pending.
- [~] `spmd.remat.policy`: `dots_saveable` versus controlled alternative
  overlays; TPU memory/throughput selection remains pending.
- [~] Logit chunk sizes 2048, 4096, and 8192 are represented by controlled
  overlays; TPU memory/throughput selection remains pending.
- [~] Host prefetch depth is now configurable through
  `runtime.prefetch_size`, with `4` and `16` A/B overlays; device-transfer
  scheduling still requires TPU evidence.
- [~] Opt-in one-shot device-memory snapshots are persisted in `metrics.jsonl`
  alongside the `.prof` artifact; TPU peak-memory evidence remains pending.
- [~] The saved-artifact comparison report now includes input wait, input
  pipeline, and captured peak-memory comparisons; TPU validation remains.
- [~] Compile deltas are now blocked when MHA/GQA cache states differ; a
  cache-matched cold/warm TPU comparison is still required.
- [~] `--clear-compilation-cache` now provides an explicit cold-cache run
  control and records the reset in the run manifest.
- [~] Constant-effective-token microbatch/gradient-accumulation overlays now
  cover 1/64, 2/32, and 4/16; TPU selection remains pending.
- [~] Compilation-cache cold versus warm control is implemented; matched TPU
  comparison remains pending.
- [~] Checkpoint interval and asynchronous-versus-synchronous checkpoint
  overlays are isolated; TPU checkpoint-overhead evidence remains pending.
- [x] Audit overlay artifact isolation before TPU allocation. The preflight
  rejects shared checkpoint/cache/profile paths; the prefetch overlays now use
  dedicated compilation caches.
- [~] Evaluate saved baseline/candidate metrics with explicit throughput, loss,
  memory, workload identity, and cache-state guards; identity now includes
  selected shards, HF revision, batch geometry, and device topology, and
  compile-sensitive comparisons can require a known matching cache state.
  Candidate acceptance can also require a real throughput or peak-memory
  improvement. TPU dispatch and stability evidence remain required.

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

**Status:** [~] Optional dispatch contracts and fallback documentation are
implemented; TPU kernel validation remains pending.

### Features

- [~] Validate Tokamax linear CE and SwiGLU only on the target TPU stack. The
  launcher now records requested loss/kernel backends and fallback policy;
  target-TPU validation remains pending.
- [~] Fix and test untied LM-head layout handling before enabling fused CE. The
  loss dispatcher records tied/untied layout and normalizes untied
  `[hidden, vocab]` weights; parity validation remains pending.
- [~] Compare native XLA, chunked CE, and Tokamax with identical inputs and
  parameters. Isolated native dense, Tokamax CE, and Tokamax kernel overlays
  plus manifest dispatch contracts are implemented; TPU evidence is pending.
- [~] Revisit scanned LLaMA layers only after the unscanned production path is
  stable. An isolated scan overlay is prepared; TPU validation is pending.
- [~] Revisit Grain only if native memmap remains the measured bottleneck. An
  isolated Grain overlay is prepared; TPU validation is pending.
- [ ] Develop FSDP all-gather overlap, sequence parallelism, and 3D mesh support as
  separate future tracks; do not mix them into PMAP production changes.

### Exit gate

- [ ] An optional optimization has a working fallback, explicit dispatch
  logging, and measured TPU benefit.
- [ ] Unsupported hardware or dependency combinations fail clearly or use a
  documented fallback.

## M8 — Export and release gate

**Goal:** Make the trained checkpoint operationally useful and reproducible.

**Status:** [~] Static release contract, audit tooling, and operational command
documentation are implemented; checkpoint/export validation remains pending.

### Features

- [~] Validate HF export with the same vocabulary size and special-token contract.
  A read-only release auditor now checks the exported JSON contract; runtime
  export validation remains pending.
- [~] Verify tied embeddings and real GQA configuration in exported metadata.
  The auditor checks both HF config and source-checkpoint metadata; TPU/export
  evidence remains pending.
- [~] Run checkpoint-to-HF parity checks on a manually selected TPU-produced
  checkpoint or approved CPU/HF validation environment outside this local TPU
  development workflow. The validator CLI now executes the previously disabled
  parity path and can persist logits/generation results as JSON; TPU evidence is
  still pending.
- [x] Publish the exact launch, resume, export, and shard-selection commands.
- [x] Publish the consolidated TPU validation runbook and artifact-return
  contract; full TPU gate execution remains pending.
- [~] Archive the final config, dependency versions, git revision, HF revision, and
  benchmark report. A static checksummed bundle builder and verifier are
  implemented, and a final readiness aggregator is available; artifact
  collection and TPU validation remain pending.

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
