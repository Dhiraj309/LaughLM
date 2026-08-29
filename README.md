# LaughLM

LaughLM is a JAX/Flax decoder-only transformer training system designed for
TPU workloads. The maintained path is a reproducible PMAP trainer using
Hugging Face pre-tokenized binary shards, explicit configuration contracts, and
Orbax checkpointing.

## Current status

- Primary trainer: `scripts/train_tpu_optimized.py`
- Primary hardware target: single TPU v5e-8 VM
- Current production config: `configs/production/laughlm_v1_127m_4b.yaml`
- Reference attention path: MHA `8/8` with SplashAttention
- Validated fresh-training candidate: GQA `8/4`
- Deferred until larger models: MaxText-style 3D tensor/sequence parallelism
- Deferred optional experiments: Tokamax kernels, Grain, scanned layers, and
  advanced FSDP execution paths

The MHA reference and GQA candidate must use separate runs. An MHA checkpoint
must not be resumed with GQA settings, or vice versa.

## Documentation handoff

- [`ROADMAP.md`](ROADMAP.md) - milestone status and acceptance gates
- [`docs/data_pipeline/ROADMAP.md`](docs/data_pipeline/ROADMAP.md) - shared
  data-pipeline and learning-integrity roadmap
- [`docs/optimization/DECISION_LOG.md`](docs/optimization/DECISION_LOG.md) -
  selected, rejected, and deferred optimization decisions
- [`docs/TPU_VALIDATION_RUNBOOK.md`](docs/TPU_VALIDATION_RUNBOOK.md) - TPU-only
  commands and artifact-return contract
- [`docs/M6_EXPERIMENT_MATRIX.md`](docs/M6_EXPERIMENT_MATRIX.md) - completed
  memory, input, and geometry experiments
- [`docs/M7_EXPERIMENT_MATRIX.md`](docs/M7_EXPERIMENT_MATRIX.md) - deferred
  kernel and execution experiments
- [`docs/M8_RELEASE_CHECKLIST.md`](docs/M8_RELEASE_CHECKLIST.md) - export and
  release gates

The FSDP 1.3B benchmark remains available at
[`docs/benchmarks/v5e_fsdp_1p3b_optimized.md`](docs/benchmarks/v5e_fsdp_1p3b_optimized.md).

Training, model code, JAX, profiling, and benchmarks must run only in the
Linux TPU environment. Static inspection and documentation work are performed
in the Windows development environment.

---

# Features

- **Decoder-only GPT architecture**
- **JAX + Flax implementation**
- **TPU-optimized mixed precision training**
- **Flexible architecture selection**
- **Pre-tokenized memory-mapped datasets**
- **Multiple attention variants**
- **Multiple FFN architectures**
- **Weight tying support**
- **Orbax checkpointing**
- **Optax optimizers**
- **Config-driven experiments**

Supported architecture features:

- MHA / MQA / GQA attention
- RoPE positional encoding
- SwiGLU / GEGLU / GELU MLP
- RMSNorm / LayerNorm
- configurable residual scaling
- multiple LR schedulers
- masked weight decay

---

# Project structure

```text
configs/                 YAML model and experiment configurations
LaughLM/config/          schema, loading, and validation
LaughLM/data/            maintained shard loading and data utilities
LaughLM/model/llama/     maintained LLaMA model implementation
LaughLM/training/        PMAP/FSDP trainers, loss, optimizer, checkpoints
LaughLM/profiling/       opt-in profiling infrastructure
scripts/                 TPU launchers and static artifact audits
docs/                    roadmap, runbooks, matrices, and handoff notes
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-org/LaughLM.git
cd LaughLM
```

Create environment:
```bash
python -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Optional integrations are not installed by the default environment because
the production PMAP path uses native XLA loss and native memmap data loading:

```bash
pip install -e ".[kernels,data]"
```

For TPU environments install JAX:

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

Configuration

Experiments are fully defined via YAML configs.

Example:

configs/test.yaml

Configuration sections include:

model architecture

optimizer

scheduler

runtime parameters

dataset sources

tokenizer settings

hardware configuration


Example snippet:
```yaml
model:
  d_model: 768
  num_layers: 12
  num_heads: 12
  vocab_size: 32000
  max_seq_len: 2048
```

---

Dataset Pipeline

LaughLM uses a pre-tokenized dataset pipeline for maximum throughput.

Training datasets are converted into binary token shards.

Advantages:

high throughput

minimal CPU overhead

memory-mapped streaming

scalable to large datasets



---

Step 1 — Dataset and tokenizer assets

The active dataset is already tokenized; tokenizer training is not required.
```bash
No tokenizer-training command is required for the active dataset.
```
Output:

tokenizer.json


---

Step 2 — Shard source

Training downloads selected pre-tokenized `.bin` shards from Hugging Face.
```bash
See the TPU validation runbook for the maintained shard-selection command.
```

For a split-aware Stage-4 corpus produced by `data_clean`, point the trainer at
the dataset repository and resolve its committed `ACTIVE.json` rather than
manually counting shard IDs:

```bash
python -u -m scripts.train_tpu_optimized \
  --config configs/production/laughlm_v1_127m_4b.yaml \
  --hf-repo-id YOUR_STAGE4_TOKEN_REPO \
  --hf-revision main \
  --stage4-active
```

This validates the Stage-4 manifest vocabulary and storage dtype, downloads
the exact train/validation shard lists, and records those paths in the run
manifest. Build both `train` and `validation` Stage-4 outputs before using it.
Output:

dataset_shard.bin

Shards contain:

uint16 token stream


---

Step 3 — Training

Run training:
```bash
python -u -m scripts.train_tpu_optimized \
  --config configs/production/laughlm_v1_127m_4b.yaml
```
Training automatically handles:

optimizer

scheduler

logging

checkpointing


Example output:

STEP   PROGRESS │ LOSS   PPL │ LR │ TOK/S │ MFU


---

Checkpointing

Checkpoints are saved using Orbax.

Default directory:

checkpoints/

Resume training automatically if checkpoints exist.


---

Benchmarking and validation

Performance measurements are TPU-only. Use
[`docs/TPU_VALIDATION_RUNBOOK.md`](docs/TPU_VALIDATION_RUNBOOK.md) and keep
each candidate isolated with its own checkpoint and compilation-cache paths.
Do not interpret a local benchmark as evidence for TPU performance.


---

Monitoring

Training logger displays:

loss

perplexity

gradient norm

tokens/sec

MFU

ETA


Example:

STEP  PROGRESS │ LOSS │ LR │ TOK/S │ MFU │ ETA


---

Optimization history

The measured optimization decisions and current priorities are maintained in
[`docs/optimization/DECISION_LOG.md`](docs/optimization/DECISION_LOG.md).



---

Development Workflow

Recommended workflow:

1. Create branch
2. Implement change
3. Run benchmark
4. Compare tokens/sec
5. Merge if improvement

Example:
```bash
git checkout -b optimize_attention
```

---

Contributing

Pull requests should include:

clear description

performance impact

benchmark results



---

License

MIT License


---

Acknowledgements

LaughLM builds on ideas from:

GPT

LLaMA

PaLM

DeepSeek

MiniCPM


and the JAX / Flax ecosystem.


---

Future work

See the root [`ROADMAP.md`](ROADMAP.md) and the optimization decision log.
Current future tracks are optional fused kernels, advanced FSDP execution,
frontier profiling, and 3D tensor/sequence parallelism for larger models.

## License

MIT License. See [`LICENSE`](LICENSE).
