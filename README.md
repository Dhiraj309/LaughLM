# LaughLM

**LaughLM** is a configuration-driven, reproducible, industry-grade decoder-only LLM training system built with JAX/Flax.

This project is **not** a notebook experiment.

It is a clean, modular, hardware-aware training framework designed for:

* Deterministic pretraining on TPU (v5e-8)
* Controlled architectural experimentation
* Reproducible fine-tuning on GPU
* Config-driven scaling
* Resume-safe checkpointing

All architectural, optimization, and runtime decisions are controlled via YAML configuration files.
Model code contains **zero hardcoded dimensions**.

---

## Core Design Philosophy

**Configuration is the single source of truth.**

* Model hyperparameters → `configs/model/`
* Training hyperparameters → `configs/training/`
* Dataset parameters → `configs/data/`
* System assumptions → `configs/system/`
* Experimental toggles → `configs/experiment/`

YAML is parsed into validated dataclasses.
The model is a pure function of structured config objects.

* No hidden defaults
* No runtime dimension inference
* No magic constants

---

## Project Structure

```text
LaughLM/
├── configs/          # All YAML configuration
├── config/           # Schema, loader, resolver
├── model/            # Transformer architecture
├── training/         # Optimizer, scheduler, trainer
├── data/             # Dataset and tokenizer logic
├── distributed/      # TPU/GPU parallelism utilities
├── utils/            # Logging, metrics, reproducibility
├── evaluation/       # Evaluation framework
├── scripts/          # Entrypoints
├── checkpoints/      # Saved training state
└── logs/             # Training logs
```

---

## Development Phases

### Phase A — Vanilla GPT Baseline

* Learned token + positional embeddings
* Multi-head scaled dot-product attention
* GELU MLP
* LayerNorm
* Adam optimizer
* Deterministic training
* Resume-safe checkpointing

**No architectural tricks.**

**Goal:** Stable, reproducible baseline.

---

### Phase B — Parity-Oriented Refinement (PAR)

Config-controlled upgrades:

* RoPE vs. learned positional embeddings
* RMSNorm vs. LayerNorm
* SwiGLU vs. GELU
* Bias removal
* Residual scaling strategies

Each change must include:

* Mathematical derivation
* Stability analysis
* Throughput comparison
* Memory comparison
* Convergence comparison

---

### Phase C — Controlled Experimentation

* Attention variants
* KV cache design
* Mixed precision strategies
* Microbatch scaling
* Architecture ablations

Experiments must be:

* Toggleable via config
* Isolated
* Baseline-compatible

---

## Hardware Targets

### Pretraining

* TPU v5e-8
* 16 GB HBM per chip
* bf16 precision

### Fine-Tuning

* 2× T4 or 2× P100 GPUs

---

## Running Training

### Train

```bash
python scripts/train.py --config configs/
```

### Resume from Checkpoint

```bash
python scripts/resume.py --checkpoint checkpoints/pretrain/step_XXXXX
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint path_to_checkpoint
```

---

## Reproducibility Guarantees

Each checkpoint stores:

* Model parameters
* Optimizer state
* Scheduler state
* RNG state
* Global step
* Full config snapshot

Training is deterministic across:

* TPU reallocation
* Resume events
* Host restarts

---

## Design Constraints

* No hardcoded dimensions inside model code
* No direct YAML access outside config loader
* No experiment logic inside baseline implementation
* All parallelism strategies must be config-driven

---

## License

Research / educational use.