# M4 run artifact audit

This audit reads only a TPU run's `run_manifest.json` and `metrics.jsonl`. It
does not execute JAX, model code, or accelerator runtime locally.

```bash
python -u scripts/audit_run_artifacts.py \
  --run-dir checkpoints/production/135M_true_h128 \
  --require-checkpoint-timings \
  --output reports/production_run_audit.json
```

`PASS` requires manifest provenance, attention/loss/cache contracts, valid
metrics JSONL, finite throughput/loss/MFU values, complete input/device timing
fields, and first-step compile-plus-execute evidence. This catches incomplete
runs before cold/warm or MHA/GQA comparisons are interpreted.
