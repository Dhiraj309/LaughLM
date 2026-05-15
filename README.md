---
tags:
- ml-intern
---
# LaughLM

A high-performance **decoder-only transformer training system** built with **JAX + Flax** and optimized for **TPU training**.

LaughLM is designed as a **research-friendly yet production-capable framework** for experimenting with modern transformer architectures while maintaining high training throughput.

The system emphasizes:

- clean modular architecture
- hardware-efficient training
- reproducible experiments
- flexible configuration
- large-scale dataset streaming
- high MFU optimization on TPUs

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

# Project Structure:
```text
.
├── configs
│   ├── gpu_test.yaml
│   └── test.yaml
├── LaughLM
│   ├── config
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── validation.py
│   ├── data
│   │   ├── domain_sampler.py
│   │   ├── memmap_loader.py
│   │   ├── shard_writer.py
│   │   ├── tokenizer.py
│   │   └── tokenizer_train.py
│   ├── model
│   │   ├── gpt.py
│   │   ├── layers
│   │   │   ├── attention.py
│   │   │   ├── mlp.py
│   │   │   ├── normalization.py
│   │   │   ├── positional.py
│   │   │   └── residual.py
│   │   ├── parameter_utils.py
│   │   └── transformer_block.py
│   ├── training
│   │   ├── checkpoint.py
│   │   ├── logger.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── trainer.py
│   │   ├── train_state.py
│   │   └── train_step.py
│   └── utils
│       └── rng.py
├── LICENSE
├── log.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── scripts
    ├── build_shard.py
    └── train_gpu_test.py
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

Step 1 — Train Tokenizer

Train a tokenizer using streaming datasets.
```bash
python -m LaughLM.data.tokenizer_train
```
Output:

tokenizer.json


---

Step 2 — Build Token Shards

Convert raw text into token shards.
```bash
python scripts/build_shard.py
```
Output:

dataset_shard.bin

Shards contain:

uint16 token stream


---

Step 3 — Training

Run training:
```bash
python scripts/train_gpu_test.py
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

Benchmarking Performance

Benchmark raw training throughput:

python scripts/benchmark_train_step.py

This measures:

compile time

step time

tokens/sec

MFU


Example output:

Compile time: 18.2s
Step time: 0.048s
Tokens/sec: 430000


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

Optimization Roadmap

LaughLM is designed to progressively reach high TPU utilization.

Target MFU:

50–60% MFU on TPU v5e

Optimization phases:

Phase	Goal

Baseline	establish benchmark
Data pipeline	remove input bottlenecks
Graph optimization	eliminate Python overhead
Kernel fusion	maximize MXU utilization
Flash attention	reduce memory traffic



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

Future Work

Planned improvements:

Flash Attention

Activation checkpointing

MoE layers

PJIT sharding

distributed training

<!-- ml-intern-provenance -->
## Generated by ML Intern

This model repository was generated by [ML Intern](https://github.com/huggingface/ml-intern), an agent for machine learning research and development on the Hugging Face Hub.

- Try ML Intern: https://smolagents-ml-intern.hf.space
- Source code: https://github.com/huggingface/ml-intern

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'dignity045/LaughLM'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

For non-causal architectures, replace `AutoModelForCausalLM` with the appropriate `AutoModel` class.
