# M3 checkpoint artifact audit

This audit is read-only with respect to checkpoint contents. It does not
import JAX or restore model state. Run it after a TPU save/resume cycle:

```bash
python -u scripts/audit_checkpoint_artifacts.py \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --expected-max-to-keep 1 \
  --require-run-manifest \
  --output reports/production_checkpoint_audit.json
```

`PASS` requires Orbax step directories and `checkpoint_metadata/step_*.json`
sidecars to contain the same steps, each sidecar's internal `step` to match its
filename, the latest checkpoint to have matching metadata, and the configured
retention limit to be respected. The optional provenance check also requires
`run_manifest.json` and `metrics.jsonl`.
