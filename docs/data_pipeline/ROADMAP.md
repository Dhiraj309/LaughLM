# LaughLM Data and Learning Integrity Roadmap

**Status:** Design baseline; implementation not started for the new integrity
features.

This roadmap covers the shared work between the `data_clean` repository and
LaughLM. It is intentionally separate from the completed PMAP/TPU optimization
roadmap in the LaughLM repository.

Status flags: `[ ]` not started, `[~]` partial/in progress, `[x]` complete,
and `[d]` intentionally deferred.

The target property is not “zero memorization.” Language models necessarily
memorize some facts and text. The target is that memorization is insufficient to
produce good results: performance must transfer to unseen documents, sources,
wording, entities, domains, and compositions.

## Repository ownership

| Responsibility | `data_clean` | LaughLM |
|---|---:|---:|
| Source snapshots and revisions | Owns | Consumes |
| File-at-a-time processing | Owns | N/A |
| Exact and near deduplication | Owns | Consumes reports |
| Split assignment and leakage control | Owns | Consumes manifests |
| Source/domain/token quotas | Owns | Validates |
| Text-to-token conversion and packing | Owns | Consumes `.bin` contract |
| Training and checkpoint state | N/A | Owns |
| Training/validation monitoring | N/A | Owns |
| Transfer and memorization evaluation | N/A | Owns |
| MoE routing and dLLM diagnostics | N/A | Future LaughLM |

Do not duplicate corpus logic in LaughLM or model-evaluation logic in
`data_clean`. Both repositories must use the shared artifact and manifest
contract defined in M0.

## Shared artifact contract

Every durable stage and final dataset must be traceable through:

```text
dataset_id
dataset_revision
pipeline_version
config_hash
document_id
source_id
source_file
content_hash
normalized_hash
dedup_cluster_id
split
domain
timestamp
tokenizer_id
token_count
shard_path
```

Logical output must be invariant to worker count, VM size, and completion
order. Those settings may change performance, but must not change accepted
documents, deduplication winners, split assignment, or token quotas.

## Milestone overview

| Status | Milestone | Owner | Outcome | TPU required |
|---|---|---|---|---|
| [~] | M0 | Both | Shared contracts and invariants | No |
| [~] | M1 | `data_clean` | Bounded file execution | No |
| [~] | M2 | `data_clean` | Provenance and resumable manifests | No |
| [~] | M3 | `data_clean` | Global deduplication and deterministic splits | No |
| [~] | M4 | `data_clean` | Decontamination and sealed evaluation | No |
| [x] | M5 | `data_clean` | Deterministic mixing and tokenization | No |
| [~] | M6 | LaughLM | Training-integrity ingestion | Small TPU gate |
| [ ] | M7 | LaughLM | Transfer and capability evaluation | TPU checkpoint gate |
| [ ] | M8 | LaughLM | Memorization and leakage audits | TPU checkpoint gate |
| [ ] | M9 | Both | Checkpoint curves and promotion gates | TPU checkpoint gate |
| [ ] | M10 | LaughLM | Architecture comparison protocol | TPU runs |
| [d] | M11 | LaughLM | MoE and dLLM extensions | Future |
| [ ] | M12 | Both | Release and operational handoff | Final TPU gate |

## M0 — Shared contracts and invariants `[~]`

- [x] Freeze the shared artifact manifest schema.
- [x] Define dataset, stage, run, source-file, shard, and document identities.
- [x] Define semantic configuration hashing and schema-version migration.
- [ ] Define failure states: discovered, downloaded, processing, written,
  verified, uploaded, committed.
- [ ] Guarantee logical-output invariance across worker counts and VM sizes.
- [x] Add cross-repository documentation links.

**Exit gate:** both repositories can consume the same sample manifest without
repository-specific interpretation.

## M1 — Bounded, resource-aware file execution `[~]`

```yaml
runtime:
  file_workers: 2
  download_workers: 1
  upload_workers: 1
  max_inflight_files: 2
  batch_rows: 4096
  min_free_disk_gb: 5.0
  local_cache_dir: "/path/to/cache"
  local_temp_dir: "/path/to/ssd"
```

- [x] Add a versioned runtime profile with worker, batch, cache, and temp-root
  controls.
- [x] Keep Stage 1 file processing bounded by `max_inflight_files`.
- [ ] Separate file workers from download and upload workers.
- [ ] Preserve strict sequential mode for deduplication-sensitive stages.
- [x] Add disk-space preflight for the configured temporary root.
- [ ] Add backpressure for download and upload queues.
- [x] Remove temporary files only after verification or committed completion.
- [x] Record Stage 1 per-source download, processing, upload, and total timings.

**Exit gate:** one-worker and multi-worker pilots produce identical logical
output without unbounded queues or full-dataset materialization.

## M2 — Provenance, checksums, and resumability `[~]`

- [ ] Record immutable source revision, path, size, and hash.
- [ ] Record input/output byte counts and checksums.
- [ ] Record accepted, rejected, duplicate, and error counts by reason.
- [ ] Keep manifest publication as the final commit marker.
- [ ] Add explicit incomplete and failed-run records.
- [ ] Resume only verified committed units.
- [ ] Add a manifest consistency auditor.
- [ ] Record code revision, dependency versions, VM profile, and config hash.

**Exit gate:** an interrupted file resumes without duplicate output or silent
loss, and a completed run is independently verifiable from HF artifacts.

## M3 — Global deduplication and deterministic splits `[~]`

- [~] Preserve the current cross-dataset exact deduplication namespace.
- [ ] Add normalized-text hashes.
- [ ] Add near-duplicate clustering with documented thresholds.
- [ ] Define source-priority rules for duplicate winners.
- [ ] Assign splits after global deduplication.
- [ ] Support train, validation, source-held-out, temporal-held-out, synthetic,
  and sealed-test split categories.
- [ ] Keep document families and near-duplicate clusters in one split.
- [ ] Make split assignment stable from document identity and split seed.
- [ ] Produce split counts and train/evaluation overlap reports.

**Exit gate:** different worker counts produce identical split IDs and no
disallowed duplicate crosses a split boundary.

## M4 — Decontamination and sealed evaluation `[~]`

- [~] Preserve the current benchmark n-gram decontamination stage.
- [ ] Freeze benchmark versions and task manifests.
- [ ] Add exact, normalized, n-gram, and near-duplicate contamination reports.
- [ ] Separate public validation from sealed test data.
- [ ] Keep sealed-test text outside the training input path.
- [ ] Add source-held-out and temporal-held-out dataset builders.
- [ ] Record contamination status per document and benchmark item.

**Exit gate:** a release manifest proves which evaluation items were excluded
from every training mixture.

## M5 — Deterministic mixing, tokenization, and packing `[x]`

- [x] Preserve deterministic source-quota mixing and exact token budgets.
- [x] Add source/domain/time quotas and exposure statistics.
- [x] Freeze tokenizer identity, revision, EOS, padding, and packing rules.
- [x] Validate token IDs against vocabulary bounds.
- [x] Select `uint16` or a wider dtype from vocabulary size.
- [x] Add resumable shard-level tokenization and upload.
- [x] Retain provenance from tokenized shards to source stages.

**Exit gate:** the same mixture config produces manifest-equivalent output
across supported VM profiles and LaughLM can validate the final `.bin` contract.

## M6 — LaughLM dataset contract and training integrity `[~]`

- [x] Validate dataset, tokenizer, dtype, shard, and split contracts before
  training.
- [x] Record source and split identity in the training run manifest.
- [x] Record unique-token and exposure statistics (opt-in evidence mode).
- [x] Log finite loss, gradient norm, parameter norm/checksum, optimizer-state
  checksum, and optimizer step.
- [x] Verify the deterministic batch index advances.
- [x] Add a small fixed-batch overfit smoke gate.
- [x] Add a resume-equivalence gate for the next batch and next loss.
- [x] Keep training and evaluation data paths explicit.

**TPU gate:** short training smoke run with manifest, loss/gradient evidence,
parameter-change evidence, and iterator-advance evidence.

## M7 — Transfer and capability evaluation `[ ]`

- [ ] Fixed-checkpoint standard held-out evaluation.
- [ ] Source-held-out, temporal-held-out, and cross-domain evaluation.
- [ ] Paraphrase, surface-perturbation, entity-renaming, and counterfactual tests.
- [ ] Compositional and multi-hop tests.
- [ ] Synthetic tests with held-out generators, entities, and seeds.
- [ ] Per-source/per-domain metrics, item counts, and confidence intervals.

**TPU gate:** one checkpoint evaluated on the complete fixed suite with
machine-readable per-item and aggregate results.

## M8 — Memorization and leakage audits `[ ]`

- [ ] Exact and near-duplicate membership checks.
- [ ] Member versus non-member loss gaps.
- [ ] Rare-string and canary extraction tests.
- [ ] Prefix completion and exact continuation audits.
- [ ] Original versus paraphrased performance comparison.
- [ ] Seen versus renamed-entity comparison.
- [ ] Source and temporal overlap reports.
- [ ] Predefined clean/overlap/uncertain/contaminated classifications.

**Exit gate:** no model is promoted using validation loss alone.

## M9 — Checkpoint curves and promotion gates `[ ]`

- [ ] Evaluate checkpoints at fixed token milestones.
- [ ] Plot quality versus tokens, TPU-hours, FLOPs, and wall-clock time.
- [ ] Track loss, transfer, synthetic, and memorization metrics together.
- [ ] Normalize comparisons by tokenizer, sequence length, evaluation budget,
  tokens, and compute.
- [ ] Add confidence intervals and stop/continue/regress/promote decisions.
- [ ] Require improvement across multiple independent suites.

**TPU gate:** one controlled campaign produces the first quality-versus-compute
report.

## M10 — Architecture comparison protocol `[ ]`

- [ ] Freeze identical data snapshots and evaluation item IDs.
- [ ] Compare equal tokens and, where relevant, equal TPU-hours/FLOPs.
- [ ] Record parameter count, active parameters, and context length.
- [ ] Separate objective loss from capability metrics.
- [ ] Use identical prompt and decoding contracts.
- [ ] Preserve the PMAP 135M baseline as the reference.

## M11 — Future MoE and dLLM extensions `[d]`

### MoE

- [ ] Expert usage, routing entropy, load balance, capacity overflow, and
  dropped-token metrics.
- [ ] Expert specialization by source/domain and routing stability.
- [ ] Generalization comparison at equal active FLOPs.

### dLLM

- [ ] Quality versus denoising steps, latency, and FLOPs.
- [ ] Masked reconstruction and downstream transfer.
- [ ] Separate dLLM objective loss from autoregressive loss.

## M12 — Release and handoff `[ ]`

- [ ] Bundle source, stage, split, mixture, tokenizer, training, checkpoint,
  and evaluation manifests.
- [ ] Verify checksums and reproducibility metadata.
- [ ] Publish known limitations and contamination caveats.
- [ ] Publish exact training, resume, evaluation, and export commands.
- [ ] Add a handoff report that prevents re-exploration of rejected approaches.

## Implementation order

M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7/M8 → M9 → M10 → M11 → M12.

Work through M5 is static/CPU/HF-data validation and does not require a TPU.
TPU allocation should begin at M6, after final dataset manifests are ready.
