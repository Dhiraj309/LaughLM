# M8 release checklist

M8 is the operational handoff. The commands below are templates for the TPU
environment and are not executed during local development.

## 1. Fresh training launch

```bash
python -u -m scripts.train_tpu_optimized \
  --config configs/v5e_pmap_true135m_production.yaml \
  --hf-repo-id LaughTaleAI/LaughLM-Tokenized-Fine \
  --hf-revision main \
  --shard-directory fineweb-edu \
  --shard-filename-prefix fineweb-edu_shard \
  --train-shard-start 0 \
  --train-shard-count 28 \
  --validation-shard-start 28 \
  --validation-shard-count 2 \
  --fresh
```

## 2. Resume

Use the same command and remove only `--fresh`. The launcher restores the
latest checkpoint and native data-iterator state from the checkpoint directory.

## 3. Hugging Face export

```bash
python -u -m LaughLM.export.export_hf \
  --config configs/v5e_pmap_true135m_production.yaml \
  --checkpoint_dir checkpoints/production/135M_true_h128 \
  --output_dir releases/laughlm-135m \
  --tokenizer_dir tokenizer
```

Export validation remains runtime work and must be performed on an approved
TPU-produced checkpoint or approved external CPU/HF validation environment.

## 4. Static release audit

```bash
python -u scripts/audit_release_contract.py \
  --config configs/v5e_pmap_true135m_production.yaml \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --export-dir releases/laughlm-135m \
  --benchmark-report reports/production_baseline.md \
  --output releases/laughlm-135m/release_audit.json
```

The audit checks vocabulary size, the LaughLM special-token contract
(`bos=1`, `eos=32000`, `pad=32000`), MHA/GQA geometry, tied/untied metadata,
export files, run provenance, dependency versions, git revision, HF dataset
revision, and benchmark-report presence. It does not load JAX or model code.

## 5. Archive contents

Build the checksummed archive after the audit passes. The command is static and
does not restore a checkpoint or import JAX:

```bash
python -u scripts/build_release_bundle.py \
  --config configs/v5e_pmap_true135m_production.yaml \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --export-dir releases/laughlm-135m \
  --audit-report releases/laughlm-135m/release_audit.json \
  --benchmark-report reports/production_baseline.md \
  --output-dir releases/laughlm-135m-bundle \
  --log train_production.log
```

Add one `--profile` option for each profiler artifact that belongs to the
release. The builder refuses an existing output directory unless `--force` is
provided, refuses source/output overlap, and never deletes files. It archives
the final config, resolved run manifest, checkpoint metadata, export directory,
release audit JSON, metrics, benchmark report, selected TPU logs/profiles, and
the exact source git revision. SHA-256 checksums and dataset provenance are
written to `release_manifest.json`.

## 6. Verify the archive

Run the verifier against the completed bundle. Keep its report outside the
bundle so it does not become an unchecksummed extra file:

```bash
python -u scripts/verify_release_bundle.py \
  --bundle-dir releases/laughlm-135m-bundle \
  --require-audit \
  --output reports/laughlm-135m-bundle-verification.json
```

Verification must report `PASS`. It detects missing or modified files, stale
files left by a forced rebuild, missing checkpoint metadata, missing export
artifacts, incomplete dependency/HF provenance, and a missing or failed
release audit.
