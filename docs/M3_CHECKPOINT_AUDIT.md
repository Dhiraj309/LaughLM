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

For a deliberate incompatible-config preflight, compare the current YAML with
the latest v3 metadata before attempting resume:

```bash
python -u scripts/audit_checkpoint_compatibility.py \
  --config configs/v5e_pmap_true135m_production.yaml \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --expected-num-devices 8 \
  --output reports/production_checkpoint_compatibility.json
```

This static comparison covers model geometry, architecture, runtime, optimizer,
scheduler, dtype, and execution-contract fields. Runtime restore remains the
authoritative check for full mesh/layout compatibility.
