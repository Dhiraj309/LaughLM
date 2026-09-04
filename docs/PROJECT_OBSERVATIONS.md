# LaughLM project observations

Recorded 2026-08-29 as a handoff from the training and export work.

## Final training direction

The selected production model is the standalone 127M-class PMAP model in
`configs/production/laughlm_v1_127m_4b.yaml`.

- `d_model=1024`, `num_layers=8`, `num_heads=8`, `num_kv_heads=4`
- GQA with fused QKV and SplashAttention block size 512
- pre-RMSNorm, RoPE, SwiGLU, tied embeddings, no bias
- model size observed in the earlier matching export: 127,222,784 parameters
- sequence length 2048
- TPU target: one v5e-8 with PMAP over 8 devices
- native TPU kernels, chunked CE with a 4096 vocabulary chunk, and
  `dots_saveable` rematerialization

The deeper and thinner 576 x 30 SmolLM2-shaped experiment was slower on the
available TPU. The 768 x 18 balanced candidate was a useful comparison, but
the final choice is the 1024 x 8 shape because it has the strongest measured
throughput and the cleanest validated training path on this hardware. This is
a throughput and stability choice, not a claim that eight layers are ideal for
every model size.

## Token and step accounting

The production file is the first cumulative 4B-token milestone of a fixed 20B
WSD schedule:

- `runtime.total_tokens=4_000_000_000`
- `scheduler.horizon_tokens=20_000_000_000`
- global batch = 8 devices x 2 microbatches x 32 accumulation x 2048 tokens
- effective batch = 1,048,576 tokens per optimizer step
- the 4B stop is 3,814 complete optimizer steps and 3,999,268,864 processed
  tokens because the next full step would exceed the target
- train shards are 0 through 15; validation shards are 16 and 17
- each binary shard contains 250M tokens, so the training region is 4B tokens

For later milestones, resume the same checkpoint directory and advance the
cumulative stop to 8B, 12B, 16B, and 20B. Keep the model, optimizer, scheduler
horizon, tokenizer, data repository, shard layout, and compilation-cache
contract unchanged. Do not start a fresh model at each milestone.

Runtime cadence is intentionally sparse enough for a long run:

- log every 100 optimizer steps
- evaluate four held-out batches every 500 steps
- checkpoint every 500 steps and retain two checkpoints
- synchronous checkpointing is enabled for durability and export safety

## Dataset contract

Training reads the already mixed and tokenized corpus from:

`LaughTaleAI/LaughLM-Tokenized-Fine/laughlm-v1/`

The current production data selection is 16 training shards and two held-out
validation shards. The training loader must not use validation shards as a
fallback. The token vocabulary is 32,064 with the LaughLM special-token
contract observed during export: BOS 1, EOS 32,000, and PAD 32,000.

## Validated training and export observations

The earlier 4B test run restored a legacy checkpoint at step 3,814. The
legacy checkpoint had no v3 metadata, so export and parity required the
explicit legacy allowance. The new production configuration uses synchronous
checkpointing and the current checkpoint metadata path; a fresh run should not
need `--allow_legacy_checkpoint`.

The legacy export completed successfully:

- 74 tensors converted to `model.safetensors`
- 127,222,784 parameters exported
- tokenizer files and Hugging Face config written
- output was loadable through Transformers

Loss parity on a held-out token window was close:

- native JAX loss: 0.8037705, perplexity 2.2339483
- Hugging Face loss: 0.8036370, perplexity 2.2336500
- loss delta: 0.0001335
- logit mean absolute difference: 0.0094415
- logit maximum absolute difference: 0.1427388

The parity result is good enough for practical export validation. Native and
Hugging Face generation produced the same general behavior. The weak or
repetitive story quality at only 4B tokens is a model/data maturity issue, not
evidence of an export mapping failure.

## Operational lessons

- `plot_metrics.py` consumes the training metrics JSONL, not a YAML config.
  Pass a file such as `checkpoints/production/laughlm_v1_127m_20b/metrics.jsonl`.
- A `PJRT_Client_Create_Args` size error indicates an incompatible JAX,
  jaxlib, and TPU plugin stack. Install a matching TPU software set before
  debugging model code.
- The literal single-VM setting `TPU_PROCESS_ADDRESSES=local` is invalid for
  this runtime. The launchers now remove it and let JAX discover the TPU.
- The transparent-hugepages warning affects startup time, not model
  correctness. It can be addressed in the TPU VM image when available.
- `torch_xla` warning about TensorFlow is an environment warning. It is not
  part of the JAX training correctness contract.
- Never compare an MHA checkpoint with GQA settings or vice versa.
- Do not use the old override workflow for production. Use the standalone
  production YAML and a dedicated checkpoint directory.

## Fresh-start command

From the LaughLM repository on the TPU VM:

```bash
python -u scripts/train_tpu_optimized.py \
  --config configs/production/laughlm_v1_127m_4b.yaml \
  --fresh
```

For export after a completed milestone:

```bash
python -u -m LaughLM.export.export_hf \
  --config configs/production/laughlm_v1_127m_4b.yaml \
  --checkpoint_dir checkpoints/production/laughlm_v1_127m_20b \
  --output_dir hf_export_laughlm_v1_4b_clean \
  --tokenizer_dir tokenizer \
  --skip_validation
```

For parity, use the same config and checkpoint, the exported directory, and a
training shard or held-out shard that is not being modified during the check.
