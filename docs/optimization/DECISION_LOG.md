# LaughLM optimization decision log

This is the handoff for future agents and chats. It records decisions already
supported by code review or TPU evidence. Do not repeat an experiment unless
the model shape, TPU software stack, workload, or implementation changed.

## Current locked direction

- Primary trainer: PMAP on a single TPU v5e-8 VM.
- Production reference: `configs/v5e_pmap_true135m_production.yaml`.
- Data: Hugging Face pre-tokenized flat `.bin` shards. Legacy tokenizer and
  domain-sampler maintenance is out of scope.
- 135M geometry: `d_model=1024`, 8 layers, sequence length 2048, 8 query
  heads, tied embeddings, SwiGLU, RoPE, pre-RMSNorm, fused QKV, and
  SplashAttention with fallback set to `error`.
- Reference release path: MHA `8/8`. Validated fresh-training candidate: GQA
  `8/4`. Never resume an MHA checkpoint with GQA settings, or vice versa.
- Selected settings: native TPU/XLA kernels, native data backend, chunked CE
  with logit chunks of 4096, `dots_saveable` rematerialization,
  microbatch/accumulation `2/32`, host prefetch `16`, persistent compilation
  cache, and asynchronous checkpoints.
- 3D tensor/sequence parallelism is deferred for larger models and is not a
  current 135M optimization target.

## Baseline to remember

Measured PMAP MHA baseline:

- `1.014M` global tokens/sec and `1.023M` device tokens/sec;
- `53.1%` non-embedding MFU, or about `65.9%` using the logits-inclusive
  estimate;
- `5.73 GB` peak device memory out of `16.91 GB`;
- about `1.034 s` total step time and `1.025 s` device step time;
- cold first compile-plus-execute about `17.3 s`, warm about `3.3 s`.

These are TPU measurements. Changes below one percent are generally not worth
another TPU allocation without new evidence.

## Decisions by cycle

### M1 - data and tokenizer boundary

Selected flat raw `uint16` token shards for the current 32,064-token
vocabulary. The loader validates shard length, token range, dtype, batch
divisibility, host assignment, and prefetch failures. Migrate the shard format
only if the vocabulary exceeds 65,535 IDs.

### M2 - architecture and configuration contract

Selected explicit architecture metadata and fail-fast validation. Equal query
and KV head counts are MHA; GQA must have fewer KV heads. This fixed earlier
mislabelled `gqa` configurations with `8/8` or `16/16` heads. Unsupported
options, dtype conflicts, and PMAP/FSDP strategy mismatches now fail before
training.

### M3 - checkpoint and resume durability

Selected host-side 64-bit token accounting, deterministic PMAP batch-index
restore, execution-contract metadata, atomic sidecars, retention audits, and
pre-restore compatibility checks. TPU save/resume, retention, incompatible
configuration, and final artifact gates passed. Exact optimizer-tree equality
after restore remains a deferred diagnostic.

### M4 - measurement and compilation

Selected separate compile, input, transfer, device-step, checkpoint, loss,
throughput, and MFU metrics. A warm compilation cache reduced first-step
compilation time but did not improve steady-state throughput. Cache state must
match before comparing compile times.

### M5 - GQA and SplashAttention

Selected real GQA `8/4` as a validated candidate, not an automatic change to
the MHA release path. Matched cold-cache TPU comparison showed approximately
`+3.87%` throughput and `-0.268 GB` peak memory with comparable loss and compile
time. Splash dispatch was confirmed with `attention_fallback: error`.

### M6 - memory, input, and geometry tuning

| Experiment | Decision | TPU result / reason |
|---|---|---|
| Splash block 256 | Rejected | About 13.6% slower |
| Splash block 1024 | Not selected | About 0.3% faster; negligible memory change |
| Remat `nothing_saveable` | Not selected | About 0.04% faster and about 0.6% lower memory; effectively neutral |
| Logit chunks 2048 | Rejected | About 1.2% slower |
| Logit chunks 8192 | Rejected | About 2.6% slower |
| Prefetch 4 | Rejected | Neutral/slightly worse |
| Prefetch 16 | Selected | About 0.20% faster and 33.6 MB lower peak memory |
| Microbatch 1 / GA 64 | Rejected | Candidate gate failed |
| Microbatch 4 / GA 16 | Rejected | About 6.1% slower; candidate gate failed |

Device execution accounts for roughly 99% of step time, so more prefetch-depth
or microbatch/accumulation sweeps need new evidence before using TPU quota.

### M7 - optional kernels and execution paths

Deferred intentionally. Tokamax linear CE, Tokamax fused kernels, native dense
CE, scanned layers, Grain, untied LM-head experiments, and advanced FSDP/3D
paths have overlays and dispatch contracts, but are not production-selected
without target-TPU dispatch, fallback, loss, stability, and improvement
evidence. Run one isolated candidate at a time if this work resumes.

### M8 - export and release

The selected MHA checkpoint passed checkpoint, HF export/parity, release audit,
bundle verification, and readiness gates. The release includes safetensors,
HF configuration/generation metadata, and the Phi-3.5 fast-tokenizer assets.
M7 experiments and future 3D/FSDP work are not release blockers.

## Tracks not to mix into 135M production

- **MaxText-style 3D parallelism:** current mesh and logical-axis code is
  scaffolding, not a validated production path. Fused QKV and Splash head
  sharding still need a deliberate TP design. Revisit for larger models or
  long context, not for the memory-comfortable 135M run.
- **FSDP / ZeRO-3:** current FSDP is GSPMD/compiler-managed parameter and state
  sharding, not proven DeepSpeed-style ZeRO-3 with explicit gather/release and
  reduce-scatter lifecycle.
- **Frontier profiling:** MFU and timing summaries exist, but granular
  parameter/optimizer/activation buckets, collective-versus-GEMM tracing, OOM
  forensics, and Perfetto/Chrome trace export are separate work. Keep them
  disabled by default and separate from training comparisons.

## Handoff rules

1. Preserve the MHA reference run and its artifacts.
2. Use fresh checkpoint, compilation-cache, and profile paths for every
   candidate overlay.
3. Change one variable per TPU experiment and keep shards, token geometry,
   cache state, and TPU topology matched.
4. Require dispatch evidence, finite/stable loss, run/checkpoint audits, and a
   real throughput or memory improvement before selecting a candidate.
5. Record the TPU command, report path, result, and keep/reject decision in
   this file or the relevant experiment matrix.
6. Never run training, model code, JAX, profiling, or benchmarks in the Windows
   development environment. Use the TPU validation runbook for Linux commands.

## Related documents

- [`ROADMAP.md`](../../ROADMAP.md) - milestone status and acceptance gates.
- [`M6_EXPERIMENT_MATRIX.md`](../M6_EXPERIMENT_MATRIX.md) - completed tuning matrix.
- [`M7_EXPERIMENT_MATRIX.md`](../M7_EXPERIMENT_MATRIX.md) - deferred optional paths.
- [`TPU_VALIDATION_RUNBOOK.md`](../TPU_VALIDATION_RUNBOOK.md) - TPU commands and artifact contract.
