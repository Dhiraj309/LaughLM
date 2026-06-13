# v5e FSDP 1.3B Optimized Benchmark

## Config

Production config:

```text
configs/v5e_fsdp_1p3b_d4_f2_optimized.yaml
```

Benchmark config used for confirmation:

```text
configs/v5e_fsdp_1p3b_d4_f2_fusedqkv_s1024_ga16_norematlogits_benchmark.yaml
```

## Hardware

```text
TPU v5e-8
mesh: data=4, fsdp=2
attention_impl: splash
```

## Model

```text
vocab_size: 32064
d_model: 2048
num_layers: 24
num_heads: 32
num_kv_heads: 32
weight_tying: true
```

## Winning Training Shape

```text
seq_len: 1024
micro_batch_per_device: 2
gradient_accumulation: 16
tokens_per_step: 131,072
```

## Winning Architecture And Loss Knobs

```text
fused_qkv: true
chunked_logits: true
logits_chunk_size: 4096
remat_logits_chunks: false
```

## Confirmation Result

60-step benchmark summary:

```text
rows:              60 / 60
steps:             1 -> 60
bottleneck:        device_step

tok/s mean:        41,249.9434
tok/s median:      41,923.0828
device tok/s mean: 41,249.9434

total step mean:   3.6357s
device step mean:  3.6357s
data wait mean:    0.0007s
host prep mean:    0.0005s
device put mean:   0.0007s
host overhead mean:0.0000s

MFU median:        42.5792%
MFU+logits median: 44.6754%
```

## Previous Reference

Earlier non-optimized 1.3B-style run:

```text
tok/s: ~33.8k
MFU:   ~37.5%
```

Approximate improvement:

```text
tok/s: +23% to +24%
MFU:   about +5 percentage points
```

## Proxy Sweep Summary

The optimized 1.3B config was selected after a smaller proxy sweep on the same v5e FSDP path.

Important proxy results:

```text
nonfused_s1024_ga8:     ~81,742 tok/s
fused_s1024_ga8:        ~87,266 tok/s
fused_s2048_ga8:        ~82,769 tok/s
fused_s1024_ga16:       ~89,495 tok/s
fused_s1024_mb4_ga8:    ~89,053 tok/s
lc8192 candidate:       ~88,770 tok/s early logs
norematlogits candidate: 93,997 tok/s
```

Proxy conclusion:

```text
fused_qkv improved throughput.
seq_len 2048 improved MFU but reduced token throughput.
micro_batch_per_device 4 did not beat micro_batch_per_device 2 with GA16.
logits_chunk_size 8192 was rejected.
remat_logits_chunks false produced the strongest proxy improvement.
```

## Decision

Keep the optimized production config as the current v5e FSDP 1.3B default candidate.

Do not change these knobs without a new benchmark:

```text
fused_qkv
seq_len
micro_batch_per_device
gradient_accumulation
logits_chunk_size
remat_logits_chunks
```

Any future replacement should beat this baseline on the same v5e-8 setup:

```text
tok/s median: 41.9k
MFU median:   42.6%
```

## Reproduction

Short confirmation run:

```bash
python -m scripts.train_tpu_fsdp \
  --config configs/v5e_fsdp_1p3b_d4_f2_fusedqkv_s1024_ga16_norematlogits_benchmark.yaml \
  --fresh \
  --max_steps 60
```

Summary command:

```bash
python -m scripts.summarize_metrics \
  --metrics checkpoints/benchmarks/fsdp_1p3b_d4_f2_fusedqkv_mb2_ga16_s1024_norematlogits \
  --skip_steps 10 \
  --last_n 50
```

Production run:

```bash
python -m scripts.train_tpu_fsdp \
  --config configs/v5e_fsdp_1p3b_d4_f2_optimized.yaml \
  --fresh
```

Resume production run:

```bash
python -m scripts.train_tpu_fsdp \
  --config configs/v5e_fsdp_1p3b_d4_f2_optimized.yaml
```
