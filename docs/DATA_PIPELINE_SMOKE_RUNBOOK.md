# LaughLM small-scale HF pipeline smoke run

This is the lowest-cost end-to-end validation path for the data pipeline and
the LaughLM training handoff. It intentionally uses one upstream Parquet file
per source and a tiny final token budget. It validates ordering and contracts;
it is not a quality estimate for the production corpus.

The pilot sources are:

- `fineweb_edu` — web text
- `finepdfs_edu` — PDF-derived text
- `finemath` — mathematics
- `stack_edu` — code

`dolma3_150b` is excluded because its current registry format is JSONL/Zstandard,
not Parquet.

## Preconditions

Run data processing from the `data_clean` repository root. Set `HF_TOKEN` with
write access to the dedicated smoke repositories. Do not reuse production
Stage-1/2/3 or final-corpus repositories.

Before Stage 3, freeze a real benchmark contract. The current draft files are
deliberately rejected by the pipeline because they contain no task IDs and no
sealed evaluation repository:

```bash
python freeze_benchmark.py \
  --benchmark-config configs/stage3/benchmarks.yaml \
  --sealed-config configs/stage3/sealed_evaluation.yaml \
  --training-repo-id YOUR_SMOKE_STAGE3_OR_FINAL_REPO \
  --benchmark-output work/smoke/benchmarks_frozen.yaml \
  --sealed-output work/smoke/sealed_evaluation_frozen.yaml \
  --manifest-output work/smoke/frozen_evaluation_manifest.json
```

Then copy the frozen benchmark/sealed references into the smoke Stage-3
configs, or create smoke-specific Stage-3 configs that reference them. Do not
replace the production draft files in place.

## Stage 1: one source file per domain

Run one command per source. `--limit-files 1` selects the first deterministic
Parquet file after the configured source patterns are sorted:

```bash
python stage1_filter.py --config configs/stage1/fineweb_edu.yaml --limit-files 1 --workers 1 --max-inflight-files 1
python stage1_filter.py --config configs/stage1/finepdfs_edu.yaml --limit-files 1 --workers 1 --max-inflight-files 1
python stage1_filter.py --config configs/stage1/finemath.yaml --limit-files 1 --workers 1 --max-inflight-files 1
python stage1_filter.py --config configs/stage1/stack_edu.yaml --limit-files 1 --workers 1 --max-inflight-files 1
```

For each source, confirm the committed manifest contains the selected source
file, output parts, row counts, checksums, and `processing_status: committed`:

```bash
python audit_manifests.py --repo-id YOUR_FINEWEB_STAGE1_SMOKE_REPO --stage stage1 --output work/smoke/fineweb_stage1_audit.json
```

Repeat the audit for the other three Stage-1 repositories.

## Stage 2: one Stage-1 source manifest per domain

```bash
python stage2_process.py --config configs/stage2/fineweb_edu.yaml --limit-sources 1
python stage2_process.py --config configs/stage2/finepdfs_edu.yaml --limit-sources 1
python stage2_process.py --config configs/stage2/finemath.yaml --limit-sources 1
python stage2_process.py --config configs/stage2/stack_edu.yaml --limit-sources 1
```

Audit the committed Stage-2 manifests and verify the shared dedup namespace is
the same across all four sources. Keep the repositories separate by source.

## Stage 3: split and decontamination smoke pass

After the frozen benchmark contract is available, build the decontamination
index once and process one Stage-2 source manifest per dataset:

```bash
python stage3_decontam.py --config configs/stage3/fineweb_edu.yaml --build-index --limit-sources 1
python stage3_decontam.py --config configs/stage3/finepdfs_edu.yaml --limit-sources 1
python stage3_decontam.py --config configs/stage3/finemath.yaml --limit-sources 1
python stage3_decontam.py --config configs/stage3/stack_edu.yaml --limit-sources 1
```

Then run the split audit for each Stage-3 repository. Confirm that training
inputs contain only `train` rows and that validation/test/held-out/sealed rows
are not copied into the training mixture.

## Stage 4: tiny mixed token corpus

Create a smoke-only Stage-4 YAML in `data_clean/configs/stage4/` with:

- a real local or HF `tokenizer.json`;
- the actual EOS token ID;
- `target_tokens: 1_000_000`;
- equal or explicitly justified source quotas across the four sources;
- `tokens_per_shard: 250_000`;
- `allowed_splits: [train]` for every source;
- a dedicated smoke output repository;
- `token_dtype: auto`.

Run:

```bash
python stage4_build.py \
  --config configs/stage4/laughlm_smoke.yaml \
  --limit-parts-per-dataset 1
```

The resulting `corpus_manifest.json` is the handoff artifact. Verify its
tokenizer hash, vocabulary size, EOS ID, packing contract, source quotas,
allowed splits, shard byte counts, and checksums before using it for training.

## LaughLM TPU handoff

Only after the data audits pass, use the existing PMAP data-smoke config on the
TPU VM. Replace the repository and shard names with the smoke corpus values:

```bash
python -u -m scripts.train_tpu_optimized \
  --config configs/production/laughlm_v1_127m_4b.yaml \
  --hf-repo-id YOUR_SMOKE_TOKEN_REPO \
  --hf-revision main \
  --shard-directory YOUR_SMOKE_SHARD_DIRECTORY \
  --shard-filename-prefix YOUR_SMOKE_SHARD_PREFIX \
  --train-shard-start 0 \
  --train-shard-count 1 \
  --validation-shard-start 1 \
  --validation-shard-count 1 \
  --max_steps 20 \
  --fresh
```

Then run the checkpoint, compatibility, run, and dataset-contract audits.
Enable `record_exposure_stats` and `training_integrity` only in a separate
evidence run; keep them disabled for throughput measurements.

This final section is TPU-only. Do not execute it on the Windows development
machine.
