# M6 TPU experiment matrix

All overlays in this matrix are partial configs. Apply exactly one overlay to
`configs/v5e_pmap_true135m_production.yaml` with
`scripts.train_tpu_optimized --override-config`.

Each overlay changes one measured variable and uses an isolated checkpoint and
compilation-cache path. Keep the dataset, model geometry, shard selection,
effective token count, and step window constant for an A/B comparison.

Before allocating TPU time, audit the isolation contract:

```bash
python -u scripts/audit_experiment_isolation.py \
  --base-config configs/v5e_pmap_true135m_production.yaml \
  --overlay-dir configs \
  --output reports/m6_experiment_isolation.json
```

| Variable | Overlay | Baseline value | Candidate value |
|---|---|---:|---:|
| Splash block size | `v5e_pmap_true135m_splash256_override.yaml` | 512 | 256 |
| Splash block size | `v5e_pmap_true135m_splash1024_override.yaml` | 512 | 1024 |
| Remat policy | `v5e_pmap_true135m_remat_nothing_override.yaml` | `dots_saveable` | `nothing_saveable` |
| Remat policy | `v5e_pmap_true135m_remat_no_batch_dims_override.yaml` | `dots_saveable` | `dots_with_no_batch_dims_saveable` |
| Logit chunk size | `v5e_pmap_true135m_logits2048_override.yaml` | 4096 | 2048 |
| Logit chunk size | `v5e_pmap_true135m_logits8192_override.yaml` | 4096 | 8192 |
| Microbatch / accumulation | `v5e_pmap_true135m_mb1_ga64_override.yaml` | 2 / 32 | 1 / 64 |
| Microbatch / accumulation | `v5e_pmap_true135m_mb4_ga16_override.yaml` | 2 / 32 | 4 / 16 |
| Checkpoint interval | `v5e_pmap_true135m_checkpoint_interval500_override.yaml` | 1000 | 500 |
| Checkpoint mode | `v5e_pmap_true135m_checkpoint_sync_override.yaml` | async | sync |

The existing prefetch, memory, MHA, and GQA overlays remain separate so those
variables are not silently combined with this matrix.

## Measurement contract

For each accepted candidate, retain the run manifest, metrics, checkpoint
timings, memory snapshot when enabled, and comparison report. A candidate is
not accepted from static inspection alone; the M6 TPU gate still requires
throughput, peak memory, compile time, input wait, loss, and checkpoint-time
evidence.

After collecting two comparable run directories, generate a static candidate
gate report:

```bash
python -u scripts/evaluate_run_candidate.py \
  --baseline checkpoints/production/135M_true_h128 \
  --candidate checkpoints/experiments/135M_prefetch4 \
  --require-memory \
  --require-cache-match \
  --require-improvement \
  --output reports/m6_prefetch4_candidate.json
```

The report blocks mismatched model/data/batch identities and records whether
cache states make compile-time deltas eligible. `--require-cache-match` turns
that state into a hard gate for compile-sensitive comparisons. Use
`--require-improvement` when deciding whether to accept an optimization; it
requires a real throughput gain or peak-memory reduction. Thresholds are
review guards, not a replacement for TPU fallback, dispatch, or stability
evidence.
