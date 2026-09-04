# Production configuration

`laughlm_v1_127m_4b.yaml` is the single standalone production configuration
for LaughLM-v1. It trains the 127M GQA model on the mixed `laughlm-v1` corpus.

It is the first cumulative 4B-token milestone of one fixed 20B WSD schedule.
For later milestones, resume from its checkpoint and increase only
`runtime.total_tokens` to `8_000_000_000`, `12_000_000_000`,
`16_000_000_000`, and finally `20_000_000_000`. Keep every other model,
optimizer, scheduler, data, and checkpoint-path setting unchanged.

## Optional 20B -> 24B continuation

`laughlm_v1_127m_24b_extension.yaml` is an explicit scheduler fork for the
reserved training shards `00080-00095`. It restores the completed 20B WSD
checkpoint, preserves optimizer state, starts at the parent final LR, and
decays to its configured continuation floor through 24B. It writes to a new
checkpoint directory and leaves the 20B run immutable.

Use it only after the 20B parent checkpoint has been audited:

```powershell
python -u scripts/train_tpu_optimized.py `
  --config configs/production/laughlm_v1_127m_24b_extension.yaml
```

Do not use `--fresh`. The first launch forks from `runtime.resume_from`; later
launches resume the latest checkpoint in the extension `checkpoint_dir`.
