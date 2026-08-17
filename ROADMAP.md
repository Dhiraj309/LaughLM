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
validation, `[x]` fully completed, `[d]` intentionally deferred and
non-blocking.

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
| [x] | M2 | Make configuration and architecture intent authoritative; MHA release and GQA candidate contracts validated | Blocking |
| [x] | M3 | Make token accounting and checkpoint resume durable; release-scoped gate passed | Blocking |
| [x] | M4 | Establish a measured PMAP performance baseline; timing instrumentation added | High |
| [x] | M5 | Validate and optimize real GQA + SplashAttention | High |
| [x] | M6 | Tune memory, input pipeline, and compilation behavior | High |
| [d] | M7 | Optional fused kernels and advanced execution paths deferred | Future |
| [x] | M8 | Release, export, and long-run operational gate | Final |

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
- [x] Report shard paths, dtype, token counts, batch shapes, process index/count,
  first-step loss, and whether any fallback was used. Run manifests, TPU logs,
  and the final run audit now provide the complete shard, batch, process, loss,
  and fallback evidence.

## M2 — Configuration and architecture contract

**Goal:** Ensure configuration values describe the model and execution that
actually run.

**Status:** [x] Dtype, MHA release, and real-GQA candidate architecture
contracts were validated on TPU; MHA remains the selected release path.

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
  explicit supported choices; label the selected 8-Q/8-KV release as MHA and
  retain the validated 8-Q/4-KV GQA configuration as an isolated candidate.

### Exit gate

- [x] The resolved configuration, model factory, checkpoint metadata, and
  startup logs share an explicit execution contract, including the validated
  MHA release and GQA candidate paths.

### TPU gate for changes in M2

- [x] Run model initialization and one training step with the production config.
- [x] Confirm the logs show the resolved dtypes, `num_heads`, `num_kv_heads`,
  actual attention implementation, and no unsupported-option fallback.

## M3 — Long-run state and checkpoint durability

**Goal:** Make 20B-token training resumable without counter overflow or stale
checkpoint metadata.

**Status:** [x] PMAP save/resume, retention, and incompatible-config gates passed
for the selected release path; remaining diagnostics are intentionally deferred.

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
- [x] Verify retention behavior with `checkpoint_max_to_keep: 1`.
  The selected async PMAP path retained the configured checkpoint count and
  passed the final artifact audit; synchronous-manager comparison remains
  deferred.

### Exit gate

- [x] A TPU run can save, stop, restart, restore, and continue with monotonic
  step and token counts.
- [x] A deliberately incompatible config is rejected before training resumes.
  Model, optimizer/scheduler, layout/dtype, and execution-contract mismatches
  now fail before restore; the static compatibility preflight now reports the
  exact mismatch paths, normalizes numeric YAML/JSON values, and permits the
  documented stage-level `runtime.total_tokens` change. The TPU negative gate
  rejected `d_model=960` against the `d_model=1024` checkpoint before training;
  manifest protection for failed attempts is now implemented.

### TPU gate for changes in M3

- [x] Run a short save/resume cycle with async checkpointing enabled.
- [d] Compare the restored step, `tokens_processed`, optimizer state, and next
  data batch against the preemption point. Step, token count, and deterministic
  next-batch index passed; exact optimizer-tree equality is deferred as a
  non-blocking diagnostic.
- [x] Test one intentionally changed dtype or architecture field and confirm
  restore rejection.

## M4 — PMAP performance baseline

**Goal:** Measure the current production path before changing kernels or mesh
behavior.

**Status:** [x] PMAP timing instrumentation and the cold/warm cache baseline
were validated on the selected production geometry.

### Features

- [x] Measure compile time separately from steady-state step time. Cold/warm
  TPU runs recorded first-step compile-plus-execute and steady-state timing.
- [x] Measure input wait, host batch preparation, and device-transfer time
  separately from model time in PMAP metrics.
- [x] Verify gradient accumulation is compiled as one scan and does not
  introduce host-side Python work per microbatch. The active PMAP train step
  uses `jax.lax.scan`, and the selected 200-step run completed with stable
  device-step timing.
- [x] Record Splash block size, rematerialization policy, logit chunk size, batch
  geometry, and effective tokens per optimizer step. The launcher now reports
  the resolved rematerialization, scan, logit, head, and cache settings;
  Splash reports its selected block size during model initialization.
- [x] Verify compilation-cache reuse on a second run. Cold versus warm TPU
  runs confirmed cache states of `0` versus `150` files and reduced first-step
  compile-plus-execute time from `17.411s` to `3.323s`; steady-state throughput
  and loss remained comparable. The static run-artifact audit checks
  provenance, timing fields, metric ordering, and value ranges.

### Exit gate

- [x] Produce a baseline report with steady-state tokens/sec, MFU, memory
  behavior, compile time, input wait percentage, and checkpoint overhead. The
  saved M4 reports and final stability audit contain the required TPU evidence.

### TPU gate for changes in M4

- [x] Run the same bounded workload twice: once cold-cache and once warm-cache.
- [x] Provide the first-step compile time, steady-state tokens/sec, loss
  stability, and input/device timing breakdown. Warm steady-state throughput
  was within `0.02%` of cold and final loss matched at `7.93394`.

## M5 — Real GQA and SplashAttention

**Goal:** Move from the current MHA-equivalent setup to validated GQA.

**Status:** [x] Matched MHA/GQA comparison, peak-memory capture, and 200-step
GQA stability validation passed. A dedicated production GQA overlay is ready;
the MHA production config remains preserved as the reference baseline.

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
- [x] Compare GQA against the current MHA-equivalent baseline for loss, memory,
  compile time, and tokens/sec. The matched 60-step cold-cache comparison
  reports `+3.87%` throughput, `-0.268 GB` peak memory, equal compile time
  within `0.047s`, and a `+0.01206` final-loss delta. A separate 200-step GQA
  run remained finite and stable at approximately `1.054M tok/s`.
- [x] Keep `attention_fallback: error` for production so an unintended XLA fallback
  cannot masquerade as a Splash benchmark.

### Exit gate

- [x] Real GQA runs with SplashAttention produce finite loss and an explicit
  performance comparison, with matched peak-memory evidence and a stable
  200-step validation window.

### TPU gate for changes in M5

- [x] Run baseline MHA-equivalent and real-GQA configurations for the same
  50-step workload.
- [x] Report attention dispatch, compile time, step time, memory outcome, loss,
  and any fallback/error. Matched MHA/GQA manifests, TPU logs, metrics, and
  memory snapshots now provide the complete bounded comparison with no
  fallback evidence.

## M6 — Memory, input, and compilation tuning

**Goal:** Optimize the measured bottleneck without changing multiple variables
at once.

**Status:** [x] Selected PMAP production settings were validated through
controlled TPU comparisons and a 200-step final stability/checkpoint gate.

### Experiment matrix

- [x] `splash_block_size`: 256, 512, and 1024 overlays are isolated. TPU
  results reject 256 (about 13.6% slower) and leave 1024 neutral (about 0.3%
  faster with negligible memory change); 512 remains the baseline.
- [x] `spmd.remat.policy`: `dots_saveable` versus controlled alternative
  overlays. TPU results leave `remat_nothing` neutral (about 0.04% faster and
  about 0.6% lower peak memory); no policy change is selected.
- [x] Logit chunk sizes 2048, 4096, and 8192 were evaluated through controlled
  overlays. TPU results reject 8192 (about 2.6% slower) and 2048 (about 1.2%
  slower); 4096 remains selected as the baseline.
- [x] Host prefetch depth is configurable through `runtime.prefetch_size`.
  Matched TPU A/B selected `16` over `4` and the former baseline: `16` measured
  +0.20% throughput, -33.6 MB peak memory, and unchanged loss.
- [x] Opt-in one-shot device-memory snapshots are persisted in `metrics.jsonl`
  alongside the `.prof` artifact; TPU candidates now provide peak-memory
  evidence.
- [x] The saved-artifact comparison report includes input wait, input
  pipeline, and captured peak-memory comparisons; the M6 candidate evaluator
  enforces workload, cache, loss, throughput, and memory guards.
- [x] Compile deltas are blocked when MHA/GQA cache states differ; matched
  cold/warm TPU comparisons completed with cache-state evidence.
- [x] `--clear-compilation-cache` provides an explicit cold-cache run control
  and records the reset in the run manifest.
- [x] Constant-effective-token microbatch/gradient-accumulation overlays
  covering 1/64, 2/32, and 4/16 were evaluated. Both alternatives failed the
  candidate gate; 4/16 measured about 6.1% slower, so 2/32 remains selected.
- [x] Compilation-cache cold versus warm control is implemented and validated
  with matched TPU comparisons.
- [x] Final async checkpointing completed at step 200 with checkpoint timing,
  compatibility, and artifact audits passing. Alternate checkpoint intervals
  and synchronous mode remain deferred because no production change is being
  selected for them.
- [x] Audit overlay artifact isolation before TPU allocation. The preflight
  rejects shared checkpoint/cache/profile paths; the prefetch overlays now use
  dedicated compilation caches.
- [x] Evaluate saved baseline/candidate metrics with explicit throughput, loss,
  memory, workload identity, and cache-state guards; identity now includes
  selected shards, HF revision, batch geometry, and device topology, and
  compile-sensitive comparisons can require a known matching cache state.
  Candidate acceptance can also require a real throughput or peak-memory
  improvement and an explicit loss-spread bound. TPU dispatch and stability
  evidence passed for the selected configuration.

### Exit gate

- [x] Select changes only when they improve steady-state throughput or memory
  while preserving loss and resume behavior.
- [x] Every accepted setting has a recorded TPU comparison.

### TPU gate for changes in M6

- [x] Run one controlled A/B experiment per variable.
- [x] Report tokens/sec, peak memory, compile time, input wait, loss, and
  checkpoint time.

## M7 — Optional fused kernels and advanced execution

**Goal:** Evaluate higher-risk optimizations only after the native PMAP path is
stable and measured.

**Status:** [d] Optional dispatch contracts and fallback documentation are
implemented; TPU kernel validation is intentionally deferred.

### Features

- [d] Validate Tokamax linear CE and SwiGLU only on the target TPU stack. The
  launcher now records requested loss/kernel backends and fallback policy;
  target-TPU validation remains pending.
- [d] Fix and test untied LM-head layout handling before enabling fused CE. The
  loss dispatcher records tied/untied layout and normalizes untied
  `[hidden, vocab]` weights; parity validation remains pending.
- [d] Compare native XLA, chunked CE, and Tokamax with identical inputs and
  parameters. Isolated native dense, Tokamax CE, and Tokamax kernel overlays
  plus manifest dispatch contracts are implemented; TPU evidence is pending.
- [d] Revisit scanned LLaMA layers only after the unscanned production path is
  stable. An isolated scan overlay is prepared; TPU validation is pending.
- [d] Revisit Grain only if native memmap remains the measured bottleneck. An
  isolated Grain overlay is prepared; TPU validation is pending.
- [d] Develop FSDP all-gather overlap, sequence parallelism, and 3D mesh support as
  separate future tracks; do not mix them into PMAP production changes.

### Exit gate

- [d] An optional optimization has a working fallback, explicit dispatch
  logging, and measured TPU benefit.
- [d] Unsupported hardware or dependency combinations fail clearly or use a
  documented fallback.

## M8 — Export and release gate

**Goal:** Make the trained checkpoint operationally useful and reproducible.

**Status:** [x] The selected MHA production checkpoint passed stability,
checkpoint, HF export/parity, release-contract, bundle, and readiness gates.

### Features

- [x] Validate HF export with the same vocabulary size and special-token
  contract. The Phi-3.5 fast tokenizer assets, HF config, generation config,
  and safetensors export passed the release audit.
- [x] Verify tied embeddings and the final selected attention configuration in
  exported metadata. The released path is MHA 8/8; GQA 8/4 remains a validated
  M5 candidate and was not promoted into the release config.
- [x] Run checkpoint-to-HF parity checks on the selected TPU-produced
  checkpoint. Logit parity passed at sequence lengths 1, 16, and 128, and
  generation validation passed; results are persisted in `hf_parity_report.json`.
- [x] Publish the exact launch, resume, export, and shard-selection commands.
- [x] Publish the consolidated TPU validation runbook and artifact-return
  contract; the selected release path completed the full TPU gate.
- [x] Archive the final config, dependency versions, git revision, HF revision,
  benchmark report, parity report, checkpoint metadata, and profiler artifact.
  The checksummed bundle and final readiness report both passed.

### Exit gate

- [x] A fresh run, resume-compatible checkpoint, export, and documented
  operational handoff are complete for the selected release path. Optional M7
  experiments and the remaining partial M3/M4 documentation items are not
  release blockers.

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
