# LaughLM TPU validation runbook

This runbook is for the Linux TPU environment. Do not execute these commands
in the local Windows development environment. The first session is bounded
validation only; full production training starts only after the gates pass.

## Gate 1: cold-cache PMAP baseline

Use the selected two training and three validation shards:

```bash
mkdir -p reports

python -u -m scripts.train_tpu_optimized \
  --config configs/v5e_pmap_true135m_production.yaml \
  --hf-repo-id LaughTaleAI/LaughLM-Tokenized-Fine \
  --hf-revision main \
  --shard-directory fineweb-edu \
  --shard-filename-prefix fineweb-edu_shard \
  --train-shard-start 0 \
  --train-shard-count 2 \
  --validation-shard-start 2 \
  --validation-shard-count 3 \
  --max_steps 60 \
  --fresh \
  --clear-compilation-cache 2>&1 | tee reports/tpu_gate1_cold.log
```

Capture the following from the log: resolved model/dtype/head geometry,
shard paths and dtype, process topology, first-step compile-plus-execute time,
steady-state throughput, loss, timing breakdown, fallback messages, and final
checkpoint completion.

## Gate 2: checkpoint and run audits

After Gate 1 completes, run these read-only audits:

```bash
python -u scripts/audit_checkpoint_artifacts.py \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --expected-max-to-keep 1 \
  --require-run-manifest \
  --output reports/tpu_gate1_checkpoint_audit.json

python -u scripts/audit_checkpoint_compatibility.py \
  --config configs/v5e_pmap_true135m_production.yaml \
  --checkpoint-dir checkpoints/production/135M_true_h128 \
  --expected-num-devices 8 \
  --output reports/tpu_gate1_compatibility.json

python -u scripts/audit_run_artifacts.py \
  --run-dir checkpoints/production/135M_true_h128 \
  --require-checkpoint-timings \
  --output reports/tpu_gate1_run_audit.json
```

Do not proceed to an expensive experiment if any required audit fails.

## Gate 3: resume and warm-cache evidence

Resume with the same config and data selection, removing only `--fresh` and
`--clear-compilation-cache`. Increase `--max_steps` so the run continues past
the saved step. Preserve both run logs and manifests before starting another
run; each run directory must remain identifiable for comparison.

Confirm that the resumed step, `tokens_processed`, optimizer state, and next
batch index are monotonic and that the cache manifest identifies the warm
candidate state.

## Gate 4: GQA comparison

Run the isolated MHA and GQA configurations with identical shard selection,
step window, TPU shape, and comparable cache state. Collect the saved metrics,
run manifests, dispatch logs, and memory profiles. Generate the comparison only
after confirming Splash dispatch and no fallback:

```bash
python -u scripts/compare_attention_runs.py \
  --mha checkpoints/experiments/135M_mha_memory \
  --gqa checkpoints/experiments/135M_gqa4_memory \
  --skip-steps 5 \
  --output reports/mha_vs_gqa.md
```

## Gate 5: M6/M7 experiments and release

Run one isolated overlay at a time. For each candidate, retain the manifest,
metrics, checkpoint timings, memory profile, and TPU log. Use the candidate
evaluator before accepting a change. Perform export, HF parity, release audit,
bundle creation, and readiness aggregation only after the final configuration
has been selected.

## What to send back

Send the Gate 1 log plus the three Gate 1 audit JSON files first. That is enough
to decide the next TPU action without spending compute on all later gates.
