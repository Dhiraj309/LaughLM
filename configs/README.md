# LaughLM configuration layout

The original flat configuration files remain unchanged for backward compatibility.
The folders below contain small YAML overlays applied on top of the validated
135M PMAP production baseline:

```text
configs/
├── testing/       # 1–10 step correctness checks
├── smoke/         # short end-to-end training runs
├── benchmarks/    # repeatable throughput/compile measurements
└── production/    # real training stages
```

Use the optimized TPU entry point with an overlay:

```bash
python -u scripts/train_tpu_optimized.py \
  --config configs/v5e_pmap_true135m_production.yaml \
  --override-config configs/production/smol_135m_4b.yaml
```

The Smol mixed corpus is already mixed in Stage 2, so the training config uses
one source with weight `1.0` and points to the tokenized `laughlm-v1` folder.
