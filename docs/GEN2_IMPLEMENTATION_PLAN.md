# LaughLM Gen-2 implementation plan

Status: research-backed implementation proposal  
Research snapshot: 2026-09-04  
LaughLM source snapshot: `1f3f5faa6c9799e242b3433a41e64d8b7c6e228d` plus the staged working tree  
Target hardware for qualification: TPU v5e-8  

This document supersedes the narrower `3D_PARALLELISM_IMPLEMENTATION_ROADMAP.md` for Gen-2 planning. That roadmap remains useful for the initial DP/FSDP/TP work, but it does not cover the current Tokamax surface, dependency migration, MoE/EP, or the GatedDeltaNet2 branch.

No TPU workload was run for this research. Every TPU command below is a manual validation command to execute after the corresponding implementation PR lands.

## Executive decision

LaughLM should own the training semantics and distributed runtime. It should adapt proven design patterns from MaxText without becoming a MaxText fork, and it should consume only public Tokamax APIs through narrow, version-checked adapters after exact-shape v5e measurements.

The recommended order is:

1. Freeze current PMAP/FSDP behavior and establish a modern, reproducibly locked dependency lane.
2. Correct FSDP batch semantics and build a shared `data x fsdp x tensor` global-array runtime behind `runtime.backend: parallel3d`.
3. Ship and qualify Dense 1.33B using native JAX and LaughLM's exact chunked cross-entropy as the reference path.
4. Add MoE as a separate model branch, first with native routing and ragged primitives, then EP8.
5. Add the 3:1 GatedDeltaNet2/Splash branch independently of MoE, first with a native JAX recurrence.
6. Promote individual Tokamax kernels only when they beat the native path on the exact v5e-8 model shapes without weakening numerical or checkpoint guarantees.

The architectural boundary is:

```text
LaughLM configuration, model semantics, training state, checkpoint contract
                              |
                    LaughLM Gen-2 runtime
            mesh + shardings + gradients + optimizer
                    /                       \
       native JAX reference paths       kernel adapters
                                           |
                               public Tokamax APIs only
```

Existing PMAP and FSDP configs remain valid. Gen-2 is opt-in. No existing run should silently acquire TP, EP, different batch accounting, a different loss, or a custom kernel.

## 1. Research basis and decision vocabulary

### Source snapshots

- LaughLM was audited from the current repository and staged working tree. The important local sources are [`training/fsdp_trainer.py`](../LaughLM/training/fsdp_trainer.py), [`training/fsdp_train_step.py`](../LaughLM/training/fsdp_train_step.py), [`distributed/mesh.py`](../LaughLM/distributed/mesh.py), [`distributed/sharding.py`](../LaughLM/distributed/sharding.py), [`utils/sharding_factory.py`](../LaughLM/utils/sharding_factory.py), [`model/llama`](../LaughLM/model/llama), [`training/loss.py`](../LaughLM/training/loss.py), and [`training/checkpoint.py`](../LaughLM/training/checkpoint.py).
- Tokamax was inspected at commit [`609307ca5a98ccd831bb534b8d283281750c17df`](https://github.com/openxla/tokamax/commit/609307ca5a98ccd831bb534b8d283281750c17df), dated 2026-09-03.
- MaxText was inspected at commit [`501f3b828793adc321dda02f524f014324395100`](https://github.com/AI-Hypercomputer/maxtext/commit/501f3b828793adc321dda02f524f014324395100), dated 2026-09-03.
- Current ecosystem behavior is grounded in the official [JAX parallel programming guide](https://docs.jax.dev/en/latest/parallel.html), [Flax sharding guide](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html), [JAX rematerialization guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html), [Orbax PyTree checkpoint guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html), [Pallas distributed TPU guide](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html), and [XProf capture guide](https://openxla.org/xprof/capturing_profiles).

Upstream `main` branches move quickly. The commit IDs above, not an unqualified `main`, are the research reference.

### Classification meanings

| Classification | Meaning for LaughLM |
|---|---|
| **USE DIRECTLY** | Depend on a documented public API. LaughLM still owns validation and integration. |
| **ADAPT** | Reimplement the pattern in LaughLM's architecture; do not import the other trainer. |
| **BENCHMARK FIRST** | Keep behind an opt-in backend until exact-shape correctness, HBM, and v5e performance gates pass. |
| **EXPERIMENTAL ONLY** | Isolated experiments may pin and import it, but production code and checkpoints must not depend on it. |
| **IGNORE** | It does not serve the current Gen-2 target or has a better native/public alternative. |

## 2. Current-state audit

### What is already worth preserving

| Area | Current capability | Assessment |
|---|---|---|
| Stable trainer | PMAP and FSDP are independently routed. `parallel3d` and `moe` are reserved and fail clearly. | Good compatibility boundary. Keep it. |
| Global-array FSDP foundation | FSDP initializes abstract state, derives logical shardings, and compiles a `jax.jit` train step with input/output shardings. | A useful base, but not yet a qualified ZeRO-3 implementation. |
| Canonical model path | PMAP and FSDP trainers use `LaughLM/model/llama`, with logical parameter annotations and activation constraints. | Extend this path only. The older `GPTModel`/`model/layers` path should not become Gen-2's second implementation. |
| Gradient accumulation | FSDP uses `lax.scan` and FP32 gradient accumulation. | Keep the device-side scan; make reduction semantics explicit. |
| Memory-efficient loss | LaughLM has exact sparse-label cross-entropy from hidden states, including an exact chunked-vocabulary custom-VJP path. It avoids materializing `[B,T,V]`. | Keep as the correctness oracle and production fallback. |
| Attention | Native JAX SDPA and TPU Splash exist; scaled-query semantics and fallback policy are explicit. | Preserve, then make the Splash wrapper TP/FSDP aware. |
| Rematerialization | Block remat, several policies, scan-over-layers, and logits-chunk remat are configurable. | Solid base; modernize policy names and profile combinations. |
| Checkpoint metadata | Checkpoints record model, scheduler, dtype, layout, token progress, and continuation provenance with strict compatibility checks. | Preserve and extend; checkpoint integrity is a release gate. |
| Profiling/metrics | Throughput, MFU, step timing, memory profiles, and run manifests exist. | Keep, but repair XProf and make unique-token accounting authoritative. |
| Data | Native memmap loading is deterministic; Grain is optional. | Do not couple the runtime project to a data-pipeline rewrite. |

### Blocking gaps and correctness risks

| Area | Evidence in current code | Required correction |
|---|---|---|
| FSDP batch semantics | `FSDPTrainer.data_replicas` and `global_batch_size` use only the `data` axis. Input sharding maps batch only to the configured `batch` rule, currently `data`. | Gen-2 batch must be sharded across `data x fsdp`; TP ranks cooperate on the same examples. Rebaseline reported tokens as unique tokens. |
| Token accounting | The current 1.3B `data=4, fsdp=2` benchmark reports 131,072 tokens/update using only four data replicas. Device token counters are `int32`. | Define one global formula and move long-run accounting to host `int64` or an explicitly enabled JAX x64 scalar. |
| Nominal 3D path | `utils/sharding_factory.py` accepts `maxtext_3d`, but silently changes requested dimensions, constructs only `(data, fsdp, tensor)`, and ignores a configured sequence factor in the resulting mesh. | Replace it with an exact `MeshPlan`; never silently rewrite a requested mesh. Deprecate the duplicate `sharding_strategy` router. |
| ICI/DCN topology | `MeshConfig.axis_sizes()` multiplies ICI and DCN factors into one value before mesh creation. | Preserve ICI and DCN vectors and use `create_hybrid_device_mesh` for multislice execution. Keep TP/EP on ICI unless measured otherwise. |
| Missing EP | There is no expert axis in the mesh/schema/runtime. | Add `expert` to the Gen-2 mesh and logical rule vocabulary, inactive at size 1 for dense/GDN. |
| Logical axes | `embed`, `heads`, and `mlp` are reused for both parameters and activations. Rule values are only `Optional[str]`. | Split parameter and activation names. Permit one logical dimension to map to a tuple of physical axes, with duplicate-axis and divisibility validation. |
| TP projections | Fused QKV is annotated `("embed", None)` and explicitly defers output partitioning. | Start TP with separate Q/K/V projections. Add a structured TP-safe fused QKV only after parity and benchmark evidence. |
| Splash sharding | GSPMD Splash uses hard-coded `P("data", None, None, None)`, requires a `data` axis, ignores FSDP/TP, and explicitly expands GQA KV heads. | Generate specs from `MeshPlan`, shard local heads over TP, batch over `data x fsdp`, and benchmark direct GQA without global KV expansion. |
| FSDP loss options | `fsdp_train_step._loss_kwargs` does not forward `loss.backend` or `tokamax_implementation`, unlike the PMAP step. | Unify loss configuration and require a logged, fail-fast implementation selection. |
| Tokamax adapters | `utils/fused_ops.py` probes speculative API names/signatures, catches broad exceptions, and permanently disables failed paths. Some calls do not match current public Tokamax signatures. | Replace with one typed adapter per public operation, an explicit supported-version range, startup capability validation, and no silent benchmark fallback. |
| LM-head TP | Chunked native CE is exact but not vocab-parallel. Tokamax linear CE expects a complete `[H,V]` weight and does not expose distributed softmax statistics. | Implement LaughLM's native vocab-parallel CE for TP. Limit Tokamax CE to TP1 unless its public API gains compatible distributed semantics. |
| Optimizer state | State sharding is derived through Flax logical partitions, but no Gen-2 structural assertion guarantees moments match parameter shards. | Derive and assert every array-valued optimizer leaf from its parameter sharding; replicate only counters/scalars. |
| Gradient reduction | Current comments rely on implicit compiler behavior and do not define exactly when DP reduction occurs relative to accumulation and clipping. | Accumulate local partial gradients, reduce once after the scan, then compute the global norm and clip. Verify collectives in HLO. |
| MoE/GDN2 | `ffn_type: moe` raises; no GatedDeltaNet model branch exists. | Build separate branches only after the shared runtime passes dense qualification. |
| XProf | `profiling/integrations/jax.py` is deliberately a no-op after a profiler-plugin ABI mismatch. | Fix the dependency pair, restore programmatic trace capture, and fail clearly when requested tracing cannot start. |
| Benchmark tooling | `benchmark_train_step.py` uses the legacy `GPTModel`, not the canonical LLaMA path. `scripts/train.py` and the TPU-specific runner hard-code datasets. | Add one synthetic, canonical Gen-2 benchmark harness. Keep dataset I/O outside kernel/runtime comparisons. |
| Dependencies | LaughLM pins JAX/JAXLIB 0.4.38, Flax 0.10.2, Optax 0.2.4, and Orbax 0.11.4; Tokamax 0.0.13 requires Python 3.12 and JAX/JAXLIB 0.11+. | A modern, separate lock is a prerequisite. Do not install unbounded `jax[tpu]` into the existing environment. |

### Interpretation of the current FSDP benchmark

The recorded v5e-8 result is still useful as a historical compiler/performance reference:

- model: current MHA 1.30B-style model, 24 x 2048, 32 Q/32 KV heads;
- mesh: `data=4, fsdp=2`;
- shape: sequence 1024, micro-batch 2, accumulation 16;
- median: 41,923 tokens/s and 42.58% non-embedding MFU.

It must be labeled **pre-correction** because the code counts only the `data` axis as unique batch replicas. Gen-2 acceptance starts from a corrected FSDP baseline; it must not claim a speedup by changing token accounting.

## 3. LaughLM vs MaxText vs Tokamax vs modern JAX

| Concern | LaughLM now | MaxText reference | Tokamax reference | Modern JAX ecosystem | Gen-2 conclusion |
|---|---|---|---|---|---|
| Ownership | Small configuration-driven trainer with strict metadata | Full large-scale training system | Kernel and autotuning library | Compiler/runtime primitives | Keep LaughLM as owner. Do not wrap or fork a whole trainer. |
| Distributed model | PMAP plus partial global-array FSDP | DP/FSDP/TP/EP and many additional axes | Individual local/distributed kernels | Auto, explicit-global, and manual-per-device modes | Build a small explicit Gen-2 runtime using JAX global arrays; reserve manual `shard_map` for kernels/collectives that require it. |
| Mesh | Flattened ICI/DCN, no EP | Separate ICI/DCN vectors and hybrid topology-aware mesh | Kernel-specific sharding | `Mesh`, `NamedSharding`, topology-aware mesh utilities | Adapt MaxText's mesh discipline with far fewer axes. |
| Logical rules | One physical axis per logical name | Activation/weight-specific names mapping to one or several physical axes | Not a trainer concern | A tensor dimension may map over a tuple of mesh axes | Adapt the rule model; keep LaughLM's schema and validation. |
| FSDP | Compiler-driven parameter/state shards, incomplete batch semantics | Per-layer layout transitions induce all-gather/reduce-scatter; batch uses data/FSDP axes | Some fused experimental collectives | Global-array sharding naturally expresses ZeRO-3-style state layout | Implement and prove FSDP through shardings and HLO. Avoid manual parameter wrappers initially. |
| TP | Config placeholders only | Column/row parallel attention and MLP with activation constraints | Attention/ragged kernels can consume sharded inputs | `jit`, explicit shardings, `shard_map` | Adapt projection layouts; start with replicated residuals over TP. |
| EP/MoE | Missing | Mature routing, sorting/ragged paths, EP all-to-all, ring variants | Public ragged dot; private experimental GMM/fused MoE | Public `lax.ragged_dot` and `lax.ragged_all_to_all` | Own routing semantics; use native JAX first; benchmark public Tokamax ragged dot. |
| Optimizer | Optax, partial logical sharding | State abstract tree follows parameter specs; optional ZeRO-1 | Not a trainer concern | PyTree shardings and `eval_shape` | Adapt structural derivation and assertions. |
| Accumulation | `lax.scan`, implicit reductions | `lax.scan`, then explicit reduced/unreduced layout transition outside scan | Not a trainer concern | Explicit sharding can describe partial/reduced values | Adapt the one-reduction-after-scan pattern. |
| Checkpoint | Strict custom Orbax wrapper and metadata | Broad modern Orbax composite/async/topology patterns | No checkpoint system | Abstract target restore; topology changes require target sharding | Keep LaughLM metadata and adapt current Orbax restore APIs. |
| Attention | JAX SDPA and direct Splash | Multiple topology-aware attention implementations | Public `dot_product_attention` with Mosaic/XLA options | JAX SDPA and Pallas Splash | Preserve native/Splash. Treat Tokamax attention as a measured alternative. |
| CE | Exact native dense and chunked implementations | Vocab tiling and distributed approaches | Public memory-efficient linear CE | XLA plus custom VJP/collectives | LaughLM owns vocab-parallel CE; Tokamax is a TP1 candidate. |
| Gated Delta | Missing | Native naive/chunked recurrence and a 3-local:1-full scheduling pattern | Private serving GDN and experimental KDA | `lax.scan`, remat, Pallas | Adapt equations and scheduling from MaxText; keep Tokamax experiments isolated. |
| Profiling | Good high-level metrics; XProf disabled | Integrated profiling and XLA flags | Public benchmark/autotune tools | `jax.profiler`, XProf, HLO dumps | Repair early, before optimization work. |
| Framework style | Flax Linen | Linen plus growing NNX use | JAX functions | Current Flax emphasizes NNX and explicit JAX sharding | Stay on Linen for Gen-2. NNX migration is unrelated risk. |

Two MaxText patterns are especially important:

1. It distinguishes activation axes from weight axes. LaughLM's single `embed` label cannot safely express an input weight dimension sharded by FSDP and an output/head dimension sharded by TP.
2. During MoE dispatch it can remove the physical `expert` axis from the dispatched token-batch dimension while assigning that axis to the expert parameter dimension. Without that role change, the compiler can generate a parameter gather/reduce-scatter where an EP all-to-all was intended.

Pallas is a kernel language, not a reason to bypass XLA by default. The JAX documentation explicitly notes that opaque Pallas kernels can prevent XLA from seeing or overlapping adjacent collectives. Every distributed Pallas candidate therefore remains profile-gated.

## 4. Adopt/adapt/benchmark/ignore decisions

### Runtime and framework

| Feature | Source | Classification | Decision |
|---|---|---|---|
| `Mesh`, `NamedSharding`, `PartitionSpec`, global `jax.Array` | JAX | **USE DIRECTLY** | Foundation of Gen-2. |
| Explicit sharding-in-types | JAX 0.11 | **USE DIRECTLY** | Target for new runtime code after the dependency gate; do not retrofit PMAP in the same PR. |
| `create_device_mesh` / `create_hybrid_device_mesh` | JAX | **USE DIRECTLY** | Use exact ICI/DCN vectors and topology-aware ordering. |
| MaxText mesh validation and ICI/DCN policy | MaxText | **ADAPT** | Copy the invariants, not its large axis set or config system. |
| MaxText logical-axis vocabulary/pattern | MaxText | **ADAPT** | Introduce a smaller LaughLM-specific v2 vocabulary. |
| MaxText whole trainer/config stack | MaxText | **IGNORE** | Rewriting LaughLM around MaxText violates scope and would discard current stable behavior. |
| MaxText optimizer-state sharding pattern | MaxText | **ADAPT** | Derive state specs from the parameter tree and assert them. |
| MaxText post-scan gradient reduction pattern | MaxText | **ADAPT** | One DP reduction after gradient accumulation; FSDP reduce-scatter through target layout. |
| Orbax abstract-target restore and async save | Orbax | **ADAPT** | Use the public API directly while preserving LaughLM's metadata and compatibility checks. |
| Flax NNX migration | Flax/MaxText | **IGNORE** | Gen-2 stays Linen. Revisit only as an independent migration after runtime stability. |
| Custom distributed Pallas collectives written by LaughLM | JAX/Pallas | **EXPERIMENTAL ONLY** | Use only if XProf proves native/compiler collectives are the bottleneck. |

### Attention, logits, and core kernels

| Feature | Public status at research snapshot | Classification | Decision |
|---|---|---|---|
| LaughLM native exact chunked CE | LaughLM public/internal stable path | **USE DIRECTLY** | Correctness oracle and fallback for all models. |
| LaughLM native vocab-parallel CE | Not implemented | **ADAPT** | Implement from distributed log-sum-exp and target-logit reductions; required for memory-efficient TP. |
| `linear_softmax_cross_entropy_loss` | Tokamax top-level public API | **BENCHMARK FIRST** | Test TP1 at `[rows,2048] x [2048,32064]`. It lacks LaughLM's `z_loss`, bias, and ignore-index contract and is not a distributed-vocab API. |
| Current JAX SplashAttention | JAX experimental Pallas API already used by LaughLM | **ADAPT** | Keep as the main TPU candidate, but replace hard-coded data-only `shard_map` and avoid global GQA KV replication. |
| `dot_product_attention` | Tokamax top-level public API | **BENCHMARK FIRST** | Compare `implementation="mosaic"` against LaughLM Splash and JAX SDPA for exact GQA forward+VJP shapes. |
| `gated_linear_unit` | Tokamax top-level public API | **IGNORE** | Native SwiGLU is simple and XLA-fusible; profile evidence is required before adding another model dependency. |
| Tokamax `layer_norm` | Tokamax top-level public API | **IGNORE** | RMSNorm is not the requested API and current native normalization is not the demonstrated bottleneck. |
| MLA | Tokamax private experimental tree | **IGNORE** | Gen-2 targets GQA/Splash, not MLA. |

### MoE and ragged operations

| Feature | Public status at research snapshot | Classification | Decision |
|---|---|---|---|
| `jax.lax.top_k` for 8 experts / Top-2 | JAX public API | **USE DIRECTLY** | Native router baseline. Eight logits are too small to justify a specialized private kernel without evidence. |
| `jax.lax.ragged_dot` | JAX public API | **USE DIRECTLY** | Native grouped-matmul reference and initial production path. |
| `jax.lax.ragged_all_to_all` | JAX public API | **USE DIRECTLY** | Initial EP dispatch/combine collective. |
| `ragged_dot` / `ragged_dot_general` | Tokamax top-level public API | **BENCHMARK FIRST** | Candidate replacement for local expert GEMMs, including `mosaic_tpu_v2`; promote only on exact balanced and skewed shapes. |
| Ragged gather/scatter/gather-reduce | Tokamax private `_src` packages | **EXPERIMENTAL ONLY** | No production imports until a supported top-level API exists. |
| GMM v2 / TGMM | Tokamax private `experimental/gmm_v2` | **EXPERIMENTAL ONLY** | Dedicated pinned experiment only; do not make checkpoints or configs depend on it. |
| Fused MoE reduce-scatter | Tokamax private `experimental/fused_moe_rs` | **EXPERIMENTAL ONLY** | The source is explicitly experimental and requires device-specific retuning; its documented strongest use case is decode. |
| TPU Top-K | Tokamax private experimental API | **IGNORE** | Use JAX Top-K for eight experts. Revisit only if router Top-K exceeds 2% of step time. |
| Ring-of-experts and advanced MaxText MoE modes | MaxText | **EXPERIMENTAL ONLY** | EP8 all-to-all is the first target. Add rings only after a measured communication bottleneck. |

### Gated Delta, KDA, and experimental kernels

| Feature | Public status at research snapshot | Classification | Decision |
|---|---|---|---|
| MaxText Qwen3/Qwen3.5 Gated Delta equations and 3:1 layer scheduling | MaxText source pattern | **ADAPT** | Implement a small LaughLM-native Linen module and recurrence tests. |
| Native token recurrence and chunked Gated Delta Rule | JAX/MaxText pattern | **ADAPT** | Token recurrence is the golden oracle; chunked scan is the first training implementation. |
| Causal Conv1D + Gated Delta Rule | Tokamax private package | **EXPERIMENTAL ONLY** | Current tests and API are prefill/decode/state-cache oriented and it is not a public training contract. |
| KDA | Tokamax private `experimental/kda` API | **EXPERIMENTAL ONLY** | It has an XLA reference and Mosaic custom VJP, but semantics and published benchmark shapes do not equal LaughLM GDN2. Require semantic parity before timing. |
| Direct custom GDN2 Pallas kernel | LaughLM | **EXPERIMENTAL ONLY** | Do not write one until the native chunked path is correct and XProf identifies a kernel-shaped bottleneck. |

### Tooling

| Feature | Source | Classification | Decision |
|---|---|---|---|
| `tokamax.benchmark` and `standardize_function` | Tokamax top-level public API | **USE DIRECTLY** | Use in the isolated kernel benchmark harness with `method="xprof_hermetic"` on TPU and `mode="forward_and_vjp"`. |
| `tokamax.autotune` / serialized `AutotuningResult` | Tokamax top-level public API | **USE DIRECTLY** | Benchmark tooling only. Never autotune during a production run; pin serialized results by device, shape, dtype, sharding, and package revision. |
| Tokamax automatic implementation selection in production | Public, but shape-dependent | **BENCHMARK FIRST** | Production must select a qualified implementation explicitly so upgrades cannot silently change numerics. |

The public boundary is deliberate. At this snapshot, Tokamax top-level exports include attention, GLU, linear CE, normalization, ragged dot, benchmarking, and autotuning. The requested GMM v2/TGMM, fused MoE, Top-K, causal GDN, KDA, MLA, and ragged gather/scatter/reduce capabilities are not top-level production APIs.

## 5. Target model contracts

All parameter counts below assume no biases, tied token embedding/LM head, and one final RMSNorm.

### Dense 1.33B

Recommended exact shape:

```yaml
model:
  vocab_size: 32064
  d_model: 2048
  intermediate_size: 6912
  num_layers: 24
  num_heads: 32
  num_kv_heads: 8
  max_seq_len: 8192
```

Count:

```text
tied embedding                           32,064 x 2,048 =    65,667,072
GQA attention/layer (Q32, KV8, Dh64)                       10,485,760
SwiGLU/layer (3 x 2,048 x 6,912)                           42,467,328
two RMSNorms/layer                                             4,096
24 decoder layers                                         1,270,972,416
final RMSNorm                                                  2,048
total                                                     1,336,641,536
```

This is not the same architecture as the existing benchmark. The current config is MHA with 32 KV heads and an implicit intermediate size of 5632. Gen-2 must add `model.intermediate_size: Optional[int]`; `None` retains the current computed LLaMA default.

### MoE 2.73B total / 0.92B active

Recommended exact shape:

```yaml
model:
  vocab_size: 32064
  d_model: 2048
  num_layers: 24
  num_heads: 32
  num_kv_heads: 8
  max_seq_len: 8192

moe:
  num_experts: 8
  experts_per_token: 2
  expert_intermediate_size: 2048
  normalize_topk_weights: true
  capacity_mode: ragged_no_drop
```

Count:

```text
eight SwiGLU experts/layer     8 x 3 x 2,048 x 2,048 = 100,663,296
router/layer                                   2,048 x 8 =      16,384
attention + norms + all experts, 24 layers                 2,668,068,864
embedding + final norm                                         65,669,120
total                                                       2,733,737,984

Top-2 active experts/layer      2 x 3 x 2,048 x 2,048 = 25,165,824
active parameters including shared model                    921,798,656
```

The count should be a unit-test contract, not a comment. Any shared expert, router bias, untied head, or different expert width changes the headline numbers.

### 3:1 GatedDeltaNet2 : Splash hybrid

Keep this as a dense-MLP architecture branch, not a MoE variant. The recommended comparable-size contract is:

```yaml
model:
  vocab_size: 32064
  d_model: 2048
  intermediate_size: 4096
  num_layers: 24
  num_heads: 32
  num_kv_heads: 8

hybrid:
  pattern: [gdn2, gdn2, gdn2, splash]

gdn2:
  conv_kernel_size: 4
  key_head_dim: 128
  value_head_dim: 128
  num_key_heads: 16
  num_value_heads: 32
  chunk_size: 64
  qk_l2_normalize: true
```

With the MaxText/Qwen3.5-style GDN projection geometry, a GDN2 module is approximately 33,718,464 parameters. Eighteen GDN2 layers, six GQA/Splash layers, and SwiGLU width 4096 produce an estimated **1,339,594,112 parameters**. A parameter-count test must freeze details such as the internal gated RMSNorm scale.

Keeping the dense width 6912 instead would produce about 1.755B parameters and confound the architecture comparison. That may be a later scaling experiment, not the first GDN2 qualification.

The layer predicate is deterministic:

```python
is_splash_layer = (layer_index + 1) % 4 == 0
```

Thus zero-based layers 3, 7, 11, 15, 19, and 23 are full Splash layers.

## 6. Recommended Gen-2 runtime architecture

### 6.1 Backend and package boundaries

Keep existing routes unchanged:

```text
runtime.backend=pmap       -> existing Trainer
runtime.backend=fsdp       -> existing FSDPTrainer
runtime.backend=parallel3d -> new MeshTrainer/Parallel3DTrainer
```

`runtime.backend=moe` can remain reserved or later become a compatibility alias for `parallel3d + architecture.ffn_type=moe`; MoE should not have a separate training loop.

Recommended new modules:

```text
LaughLM/distributed/mesh_plan.py       exact ICI/DCN mesh and role validation
LaughLM/distributed/axis_rules_v2.py   logical -> physical normalization
LaughLM/distributed/sharding_plan.py   model/state/input sharding trees
LaughLM/training/mesh_trainer.py       shared Gen-2 orchestration
LaughLM/training/mesh_train_step.py    accumulation/reduction/update semantics
LaughLM/kernels/attention.py           native/Splash/Tokamax typed adapters
LaughLM/kernels/linear_ce.py           native chunked/vocab-parallel/Tokamax
LaughLM/kernels/ragged_dot.py          native/Tokamax public adapters
LaughLM/model/llama/moe.py             router, dispatch, experts, combine
LaughLM/model/llama/gdn2.py            GDN2 projections, conv, recurrence
```

Do not duplicate model equations in the trainer. Do not route Gen-2 through the legacy `GPTModel` path.

### 6.2 Physical mesh

Use a deliberately small axis set:

```text
(data, fsdp, tensor, expert)
```

- `data`: replicated model shards over unique data.
- `fsdp`: parameter/gradient/optimizer shards whose ranks also receive unique batch shards.
- `tensor`: ranks cooperate on the same examples and shard heads/MLP/vocab.
- `expert`: size 1 for dense/GDN; later maps experts and routed traffic for MoE.

Do not add a physical sequence or pipeline axis in the first release. Sequence-parallel activation layouts may reuse the TP group later. Pipeline parallelism solves a different scale regime.

Keep ICI and DCN separately:

```yaml
spmd:
  mesh:
    ici: {data: 2, fsdp: 2, tensor: 2, expert: 1}
    dcn: {data: 1, fsdp: 1, tensor: 1, expert: 1}
```

Validation rules:

1. ICI product equals devices per slice; DCN product equals slice count.
2. At most one explicitly supported `-1` may be resolved in each vector.
3. No requested positive size is silently changed.
4. TP and EP are ICI-only by default; DCN TP/EP require a dedicated benchmark waiver.
5. Model dimensions must be divisible before device initialization.
6. Every resolved mesh and device coordinate map is written to the run manifest.

### 6.3 Logical-axis vocabulary

Use separate activation and parameter names:

```text
activations:
  activation_batch, activation_length, activation_embed,
  activation_q_heads, activation_kv_heads, activation_mlp,
  activation_vocab, activation_expert

parameters:
  param_embed_in, param_embed_out, q_heads, kv_heads,
  head_dim, mlp, vocab, expert, layers, norm
```

Rule values become `str | list[str] | null`. Example dense rules:

```yaml
spmd:
  axis_rules_version: 2
  axis_rules:
    activation_batch: [data, fsdp]
    activation_length: null
    activation_embed: null
    activation_q_heads: tensor
    activation_kv_heads: tensor
    activation_mlp: tensor
    activation_vocab: tensor
    param_embed_in: fsdp
    param_embed_out: fsdp
    q_heads: tensor
    kv_heads: tensor
    mlp: tensor
    vocab: tensor
    expert: null
    layers: null
    norm: null
```

Legacy axis rules continue through a deterministic compatibility translator used only by `pmap` and existing `fsdp`. The Gen-2 backend requires v2 rules and fails on an unknown or ambiguous name.

### 6.4 FSDP / ZeRO-3-style behavior

The intended semantics are:

1. Persistent parameters, gradients, and Adam moment arrays are partitioned over FSDP-compatible dimensions.
2. Batch activations are partitioned over `data x fsdp`, so each FSDP rank processes unique examples.
3. At each layer contraction, a parameter layout transition lets XLA issue the required all-gather just in time.
4. Backward output is constrained to the persistent parameter sharding, inducing reduce-scatter of gradients.
5. Data-parallel replicas all-reduce matching parameter shards after accumulation.

This is ZeRO-3-style only after all of the following are demonstrated:

- no persistent full parameter or optimizer copies;
- per-device state HBM scales approximately with the FSDP factor;
- HLO/XProf shows layer-scoped all-gather and reduce-scatter rather than whole-model gathers;
- a tiny update matches the unsharded reference;
- checkpoint restore reconstructs the same global arrays.

Do not initially write manual all-gather wrappers. Add them only if compiler inspection proves the layout constraints are insufficient.

### 6.5 Tensor parallelism

Use classic column/row parallelism first, with residual activations replicated over TP. This is simpler to validate than introducing sequence parallelism simultaneously.

| Operation | Parameter shape | Persistent sharding | Output behavior |
|---|---:|---|---|
| Q projection | `[D,QH,Dh]` | `P(fsdp,tensor,None)` | Q heads local to TP rank |
| K/V projection | `[D,KVH,Dh]` | `P(fsdp,tensor,None)` | KV heads local; no full KV repeat |
| O projection | `[QH,Dh,D]` | `P(tensor,None,fsdp)` | reduce over TP, return replicated residual layout |
| Gate/up | `[D,M]` | `P(fsdp,tensor)` | MLP width local to TP rank |
| Down | `[M,D]` | `P(tensor,fsdp)` | reduce over TP, return replicated residual layout |
| Tied embedding | `[V,D]` | candidate `P(tensor,fsdp)` | shared with vocab-parallel LM head |

Q32/KV8 is divisible by TP2, TP4, and TP8. Each candidate still needs exact v5e measurement.

Start with `fused_qkv: false` for TP. A later structured fused kernel may store `[D, Q_plus_2KV, Dh]` with a documented head-group partition, but only if it preserves Q/K/V conversion and beats separate projections.

Sequence parallelism is a later layout mode:

- all-gather the residual before column-parallel projections;
- reduce-scatter after row-parallel projections;
- keep normalization inputs sharded when legal.

It is not a fifth mesh factor in the initial runtime.

### 6.6 Optimizer-state sharding

- Build the abstract parameter tree first.
- Initialize Optax with `jax.eval_shape`.
- For each parameter-associated moment, use the exact parameter `NamedSharding`.
- Replicate scalar `count`, scheduler state, and small loss-scale values.
- Assert at startup that no array-valued moment unexpectedly uses `P()` when its parameter is sharded.
- Save the global logical optimizer state through Orbax, not process-local shards as a bespoke format.

Reference qualification uses FP32 parameters and FP32 moments to isolate sharding correctness. Existing `mu_dtype=bfloat16` remains supported, but it is a separate convergence/performance experiment. Do not combine dtype changes with first TP or EP qualification.

### 6.7 Gradient accumulation and reduction

Define one batch contract:

```text
unique_batch_replicas = data * fsdp                       # dense/GDN
unique_batch_replicas = data * fsdp * expert              # MoE before dispatch
tokens_per_update = micro_batch_per_replica
                  * unique_batch_replicas
                  * sequence_length
                  * gradient_accumulation
```

The tensor factor never multiplies unique samples.

Train-step order:

1. Split the loaded global batch into accumulation microbatches.
2. Compute weighted loss sums and local gradients in `lax.scan`.
3. Accumulate gradient buffers in FP32 and token/example denominators in FP32/int64-safe host state.
4. Outside the scan, transition data-replica partials to reduced shardings exactly once.
5. Normalize by the global count of non-ignored target tokens.
6. Compute the global norm from reduced, FSDP-sharded gradients.
7. Clip once, update once, and increment step/tokens once.

Tests must detect a DP all-reduce inside the accumulation scan and reject the lowered program unless an explicit exception is documented.

### 6.8 Rematerialization

Use public `jax.checkpoint_policies`; JAX considers custom policy callables an internal contract.

Initial benchmark set:

- no remat;
- full decoder-block remat;
- `dots_with_no_batch_dims_saveable`;
- named save points for block input, attention output, MLP input, and GDN chunk state;
- independent LM-logits-chunk remat on/off.

Keep `scan_layers: false` for initial TP qualification. Layer scan can reduce compile size but can also constrain scheduling. Enable it only when compile time or executable size is the measured blocker. GDN's inner chunk scan and outer layer scan must be benchmarked separately to avoid accidental nested full remat.

### 6.9 Checkpoint and resume contract

Preserve LaughLM's strict metadata philosophy and move to a canonical global-state contract:

```text
state:
  params, opt_state, step, tokens_processed, rng
data:
  source/shard cursor, process-independent sample position, packing state
architecture:
  model family/version, MoE or GDN2 fields, exact layer pattern
layout:
  logical axis vocabulary version, mesh plan, sharding tree digest
kernels:
  requested and resolved implementation, Tokamax version/revision, tuning digest
environment:
  Python/JAX/JAXLIB/libtpu/Flax/Optax/Orbax versions and container digest
```

Restore uses an abstract target PyTree carrying the new mesh shardings. Orbax explicitly requires target sharding when topology changes. The format must support:

- same-layout resume;
- `data=4,fsdp=2,tp=1` to `data=2,fsdp=2,tp=2` restore for the same model;
- synchronous and asynchronous save;
- interrupted async-save recovery without selecting an incomplete step;
- native vs qualified-kernel resume with identical mathematical state;
- canonical gather/export for Hugging Face without assuming every host owns full parameters.

The first Gen-2 checkpoint version may refuse cross-layout restore, but `parallel3d` cannot become production until the topology-change test passes. Shape-changing model surgery remains a separate, explicit tool.

### 6.10 Splash/GQA and attention selection

The initial production candidate is LaughLM's direct JAX Splash path, adapted as follows:

- global Q/K/V layout `[batch, length, heads, head_dim]`;
- batch sharded over `data x fsdp` (and `expert` for MoE shared blocks);
- local Q/KV heads sharded over `tensor`;
- one Splash call per local shard through a generated `shard_map` contract;
- causal and packed-segment masks tested at lengths 2048 and 8192;
- no silent fallback in benchmark or production configs.

Current code repeats KV heads from 8 to 32 at the Splash boundary. The Gen-2 benchmark must compare:

1. local repeat after TP sharding;
2. a direct GQA-capable JAX path if available in the pinned version;
3. public Tokamax `dot_product_attention` with Q32/KV8.

The winner is selected by full forward+VJP step behavior, not attention-only forward latency.

### 6.11 LM head and cross-entropy

Keep native exact chunked CE for TP1. Implement native vocab-parallel CE for TP>1:

1. Compute each rank's local logits for its vocabulary interval.
2. Reduce the local maximum across TP for a global stable maximum.
3. Reduce local `sum(exp(logit-global_max))` across TP.
4. Select the target logit only on its owning rank and reduce it across TP.
5. Apply ignore masking and normalize by global valid-token count.
6. Compute z-loss from the same global `logsumexp`, preserving exact LaughLM semantics.

No `[B,T,V]` global tensor may be materialized. Test against dense native CE for random labels, ignored labels, tied and untied heads, z-loss on/off, TP1/2/4/8, and vocabulary sizes not naturally aligned to tile size.

Tokamax linear CE is a TP1 candidate. Gathering a TP-sharded LM head merely to call it defeats the Gen-2 memory goal and is not an acceptable production integration.

### 6.12 MoE routing, EP, and ragged GEMM

LaughLM owns these semantics:

- FP32 router logits;
- `jax.lax.top_k(..., k=2)`;
- normalized Top-2 weights;
- deterministic tie behavior;
- load-balancing auxiliary loss and router z-loss;
- stable token/expert assignment order;
- no dropped tokens in the baseline;
- explicit routing statistics: tokens/expert, max/mean, entropy, overflow/drop count, all-to-all bytes.

Implementation ladder:

1. Tiny dense reference: one-hot dispatch and all-expert `einsum`, used only for tests.
2. Single-device ragged path: flatten Top-2 assignments, stable-sort by expert, native `jax.lax.ragged_dot`, inverse permutation, weighted combine.
3. EP path: route assignments with native `jax.lax.ragged_all_to_all`, run local expert ragged GEMMs, return outputs, inverse-permute, and combine.
4. Tokamax public ragged-dot backend, only after exact-shape qualification.

For EP8 on v5e-8, one expert resides on each rank. The `expert` physical axis has two roles:

- shared layers: it may participate in unique batch sharding, and optionally FSDP-style sharding of shared weights;
- expert region: remove `expert` from the dispatched token-row sharding and map it to the expert parameter dimension.

Benchmark both replicated shared weights and `expert_as_fsdp_for_shared=true`. Do not assume the more memory-efficient layout is faster.

### 6.13 GatedDeltaNet2 branch

Adapt the following equations, not the entire Qwen/MaxText model:

```text
(q_raw, k_raw, v_raw, z) = Linear_qkvz(x)
(b, a)                   = Linear_ba(x)
(q, k, v)                = silu(depthwise_causal_conv1d(q_raw, k_raw, v_raw))
beta                     = sigmoid(b)
g                        = -exp(A_log) * softplus(a + dt_bias)
o                        = gated_delta_rule(q, k, v, g, beta)
y                        = RMSNorm(o) * silu(z)
output                   = Linear_out(y)
```

Correctness ladder:

1. FP32 per-token recurrence, including initial/final state.
2. Chunked parallel training rule at chunk size 64.
3. Packed-document segment reset tests so state never crosses document boundaries.
4. BF16 compute with FP32 recurrent accumulation where required.
5. TP head sharding and remat.
6. Only then compare Tokamax KDA or causal GDN experiments.

KDA is not assumed to be semantically interchangeable with GDN2. An adapter must first prove output, final-state, and VJP parity on tiny sequences. A fast but different recurrence is a different architecture and cannot replace GDN2 under the same checkpoint identifier.

## 7. Configuration and usage contract

### Dense Gen-2 example

```yaml
model:
  vocab_size: 32064
  d_model: 2048
  intermediate_size: 6912
  num_layers: 24
  num_heads: 32
  num_kv_heads: 8
  max_seq_len: 8192

architecture:
  attention_variant: gqa
  attention_impl: splash
  attention_fallback: error
  fused_qkv: false
  ffn_type: swiglu

runtime:
  backend: parallel3d
  seq_len: 2048
  micro_batch_per_replica: 1
  gradient_accumulation: 16

loss:
  backend: native
  distributed_vocab: true
  logits_chunk_size: 4096
  z_loss: 1.0e-4

spmd:
  axis_rules_version: 2
  mesh:
    ici: {data: 2, fsdp: 2, tensor: 2, expert: 1}
    dcn: {data: 1, fsdp: 1, tensor: 1, expert: 1}
```

This shape has four unique batch replicas, TP2, and 131,072 tokens/update:

```text
1 microbatch x 4 replicas x 2048 tokens x 16 accumulation = 131,072
```

### MoE EP8 example

```yaml
architecture:
  attention_variant: gqa
  attention_impl: splash
  ffn_type: moe

moe:
  num_experts: 8
  experts_per_token: 2
  expert_intermediate_size: 2048
  capacity_mode: ragged_no_drop
  router_dtype: float32
  grouped_matmul_backend: native_ragged_dot
  dispatch_backend: native_ragged_all_to_all
  expert_as_fsdp_for_shared: false

runtime:
  backend: parallel3d
  seq_len: 2048
  micro_batch_per_replica: 1
  gradient_accumulation: 8

spmd:
  mesh:
    ici: {data: 1, fsdp: 1, tensor: 1, expert: 8}
    dcn: {data: 1, fsdp: 1, tensor: 1, expert: 1}
```

### GDN2/Splash example

```yaml
architecture:
  ffn_type: swiglu
  attention_impl: splash

hybrid:
  pattern: [gdn2, gdn2, gdn2, splash]

gdn2:
  conv_kernel_size: 4
  key_head_dim: 128
  value_head_dim: 128
  num_key_heads: 16
  num_value_heads: 32
  chunk_size: 64
  qk_l2_normalize: true
  implementation: native_chunked

runtime:
  backend: parallel3d
```

Every kernel-bearing config records both `requested_implementation` and `resolved_implementation`. `fallback: error` is mandatory in benchmark and production qualification configs.

## 8. Phased roadmap and dependency order

| Phase | Scope | Depends on | Exit gate |
|---|---|---|---|
| 0. Freeze | Historical benchmark, unique-token definitions, deterministic tiny oracle | None | Existing PMAP/FSDP behavior and checkpoints are reproducible. |
| 1. Runtime modernization | Modern lock, XProf, canonical benchmark harness | Phase 0 | New environment passes all legacy tests and restores a legacy checkpoint. |
| 2. Shared distributed foundation | Correct FSDP batch, mesh plan, v2 axes, optimizer/gradient contracts, checkpoint v4 | Phase 1 | TP1 Gen-2 matches single-device native and resumes across layouts. |
| 3. Dense TP | TP MLP, TP GQA/Splash, distributed CE, Dense 1.33B configs | Phase 2 | At least one v5e-8 DP/FSDP/TP mesh passes correctness, HBM, and performance gates. |
| 4. MoE reference | Router, tiny dense oracle, native ragged single-device | Phase 3 | MoE outputs/VJPs and exact 2.734B/0.922B counts pass. |
| 5. EP8 | EP mesh role switch, ragged all-to-all, checkpoint/metrics | Phase 4 | EP8 has no dropped tokens, stable routing, correct resume, and acceptable communication overhead. |
| 6. Tokamax MoE candidate | Public ragged-dot adapter and pinned tuning | Phase 5 | It wins exact-shape v5e trials or stays opt-in. Private GMM/fused paths remain experiments. |
| 7. GDN2 native | GDN2 oracle/chunked module, packed resets, 3:1 pattern | Phase 3, independent of 4-6 | Native GDN2 has forward/VJP/state parity and full-step checkpoint integrity. |
| 8. GDN2 distributed/experimental | TP/remat/profile; optional KDA experiments | Phase 7 | Qualified native hybrid config; experiments do not enter production imports. |
| 9. Release | Long smoke, resume, docs, runbooks, rollback | Relevant branch phase | Stable defaults unchanged; Gen-2 configs carry explicit experimental/production status. |

Do not start MoE and GDN2 implementation in the same PR. They share the runtime but have unrelated numerical and kernel risks.

## 9. Ordered, independently mergeable PR sequence

| PR | Change | Required tests / artifact | Default behavior |
|---:|---|---|---|
| 1 | Add legacy and Gen-2 environment inputs/locks; CI compatibility matrix. | `pip check`, package-version manifest, full current test suite on both CPU lanes. | Unchanged. |
| 2 | Repair programmatic XProf integration and add canonical synthetic benchmark/compile/HLO tools. | A CPU no-op test plus a manual v5e trace artifact; benchmark uses `model/llama`. | Profiling remains off. |
| 3 | Define `micro_batch_per_replica`, unique token accounting, and host-safe counters. | Sample-ID coverage and accounting tests; historical benchmark labeled pre-correction. | Legacy semantics retained for legacy backends/configs. |
| 4 | Add `MeshPlan`, separate ICI/DCN, `expert`, exact product checks, and topology manifest. | Fake-device mesh tests and v5e compile-only matrix. | New code unused. |
| 5 | Add logical-axis v2 and split model parameter/activation annotations; add optional `intermediate_size`. | Legacy translator tests, duplicate-axis rejection, exact parameter-count tests. | Existing configs translate identically. |
| 6 | Correct FSDP batch sharding and derive/assert optimizer shardings. | Tiny single-device vs FSDP loss/gradient/update; no duplicated sample IDs; HBM/HLO artifact. | Existing `fsdp` migration is explicit and documented. |
| 7 | Add checkpoint v4 global-state restore and cross-layout target sharding. | Same-layout, async interruption, and FSDP-layout-change resume equivalence. | Old checkpoint restore remains supported. |
| 8 | Register `parallel3d` using shared mesh trainer with TP1 parity. | PMAP/FSDP/parallel3d TP1 equivalence and 3-step resume. | Opt-in backend only. |
| 9 | Add column/row TP for SwiGLU; residual replicated over TP. | TP1/2/4 output, VJP, update, sharding-tree, and collective tests. | TP1 unchanged. |
| 10 | Add TP GQA and generated Splash `shard_map`; keep fused QKV off under TP. | Q32/KV8 TP1/2/4/8 parity, packed mask tests, no unintended KV all-gather. | Existing attention unchanged. |
| 11 | Add native vocab-parallel CE and unify loss options across trainers. | Dense/chunked/distributed CE and z-loss parity, ignored labels, tied head, no global logits buffer. | Native remains default. |
| 12 | Add typed public Tokamax CE/attention adapters and kernel benchmark suite. Quarantine legacy `fused_ops` probes. | Version/capability tests; exact v5e result table; fallback-error behavior. | Tokamax off. |
| 13 | Add Dense 1.33B configs, sharding overlays, acceptance report, and HF export path. | Parameter count, v5e-8 benchmark matrix, checkpoint/resume/export. | Gen-2 config marked candidate until report passes. |
| 14 | Add MoE schema, router, dense oracle, load-balance losses, and single-device ragged path. | Forced-routing golden tests, finite differences/VJP, no-drop combine, exact counts. | MoE off. |
| 15 | Add EP role-aware shardings and native ragged all-to-all dispatch/combine. | EP1/2/4/8 parity, skewed routing, communication metrics, EP checkpoint restore. | EP1 default in tests. |
| 16 | Add public Tokamax ragged-dot candidate and pinned autotuning records. | Balanced/skewed exact-shape v5e trials and three-repeat stability. | Native ragged dot remains default until promotion. |
| 17 | Add native GDN2 token oracle, chunked rule, causal depthwise conv, and 3:1 layer pattern. | Forward/state/VJP/packed reset/parameter-count tests. | GDN2 off. |
| 18 | Add GDN2 TP/remat and isolated private-Tokamax experiment harness. | v5e hybrid full-step matrix; private imports statically forbidden outside `scripts/experiments`. | Native GDN2 remains default. |
| 19 | Production hardening and runbooks. | Multi-checkpoint resume soak, fault injection, XProf review, release checklist. | PMAP and current FSDP still available. |

Each PR should contain its rollback switch and should be mergeable with all new behavior disabled.

## 10. v5e-8 benchmark and validation plan

### 10.1 CLI contract to add in PR 2

The roadmap assumes these stable commands are implemented before runtime optimization begins:

```text
python -m scripts.validate_gen2       numerical and sharding validation
python -m scripts.compile_gen2        lower/compile and emit HLO without a training loop
python -m scripts.benchmark_gen2      synthetic canonical full-step benchmark
python -m scripts.benchmark_kernels   exact-shape native/Tokamax kernel suite
python -m scripts.checkpoint_gen2     save/restore/resume equivalence harness
```

All commands must create a new output directory and refuse to overwrite it. They must never use a production checkpoint directory. Synthetic token IDs and forced MoE routing are seeded and recorded.

### 10.2 Environment verification

Run on the v5e host from the repository root after PR 1:

```bash
python -m pip install --require-hashes \
  -r requirements/locks/gen2-tpu-v5e-py312.txt
python -m pip check
python -m scripts.capture_environment \
  --output artifacts/gen2/environment/v5e8.json
python - <<'PY'
import importlib.metadata as m
import jax
import tokamax

for name in (
    "jax", "jaxlib", "flax", "optax", "orbax-checkpoint",
    "tokamax", "xprof", "grain", "pydantic", "libtpu",
):
    print(f"{name}=={m.version(name)}")
print(jax.devices())
print("tokamax_revision=", getattr(tokamax, "__git_revision__", None))
assert jax.device_count() == 8
assert all("TPU" in d.device_kind.upper() for d in jax.devices())
PY
```

### 10.3 Preserve the historical baseline

This is the only command here that uses an existing script and `--fresh`. Confirm that its benchmark checkpoint directory contains nothing valuable before running, because `--fresh` deletes that directory.

```bash
python -m scripts.train_tpu_fsdp \
  --config configs/v5e_fsdp_1p3b_d4_f2_fusedqkv_s1024_ga16_norematlogits_benchmark.yaml \
  --fresh \
  --max_steps 60
python -m scripts.summarize_metrics \
  --metrics checkpoints/benchmarks/fsdp_1p3b_d4_f2_fusedqkv_mb2_ga16_s1024_norematlogits \
  --skip_steps 10 \
  --last_n 50
```

Record it as `legacy/pre-correction`; do not compare its token rate directly to corrected unique-token results.

### 10.4 Dense full-step mesh matrix

PR 13 should add these complete configs. All sequence-2048 rows use micro-batch 1 and exactly 131,072 unique tokens/update.

| ID | Config | Mesh `(D,F,T,E)` | Accumulation | Purpose |
|---|---|---:|---:|---|
| D0 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d4_f2_t1_s2048.yaml` | `(4,2,1,1)` | 8 | Corrected FSDP baseline |
| D1 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d1_f8_t1_s2048.yaml` | `(1,8,1,1)` | 8 | Memory-first FSDP |
| D2 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d2_f2_t2_s2048.yaml` | `(2,2,2,1)` | 16 | Balanced 3D candidate |
| D3 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d1_f4_t2_s2048.yaml` | `(1,4,2,1)` | 16 | Memory-first TP2 |
| D4 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d1_f2_t4_s2048.yaml` | `(1,2,4,1)` | 32 | TP4 candidate |
| D5 | `configs/gen2/benchmarks/dense_1_33b_v5e8_d1_f1_t8_s2048.yaml` | `(1,1,8,1)` | 64 | TP scaling diagnostic, not presumed winner |

Compile every row first:

```bash
for cfg in configs/gen2/benchmarks/dense_1_33b_v5e8_*_s2048.yaml; do
  name=$(basename "${cfg}" .yaml)
  python -m scripts.compile_gen2 \
    --config "${cfg}" \
    --output-dir "artifacts/gen2/compile/${name}"
done
```

Then run three measured trials per row. The harness performs 10 unmeasured warm-up steps, 5 XProf steps, and 50 measured steps:

```bash
for cfg in configs/gen2/benchmarks/dense_1_33b_v5e8_*_s2048.yaml; do
  name=$(basename "${cfg}" .yaml)
  for trial in 1 2 3; do
    python -m scripts.benchmark_gen2 \
      --config "${cfg}" \
      --synthetic-data \
      --seed 20260904 \
      --warmup-steps 10 \
      --xprof-steps 5 \
      --measure-steps 50 \
      --output-dir "artifacts/gen2/dense/${name}/trial-${trial}"
  done
done
```

Repeat only D0-D4 at sequence 8192 with the corresponding configs. To retain 131,072 tokens/update at micro-batch 1, use accumulation 2 for `D*F=8`, 4 for `D*F=4`, and 8 for `D*F=2`.

### 10.5 Dense correctness and checkpoint commands

```bash
python -m scripts.validate_gen2 \
  --config configs/gen2/validation/dense_tiny_parallel3d.yaml \
  --compare-meshes single,d2_f2_t1,d1_f2_t2 \
  --steps 3 \
  --check-loss \
  --check-gradients \
  --check-updates \
  --check-sample-coverage \
  --output-dir artifacts/gen2/validation/dense-tiny

python -m scripts.checkpoint_gen2 \
  --config configs/gen2/validation/dense_tiny_parallel3d.yaml \
  --save-mesh d4_f2_t1 \
  --restore-mesh d2_f2_t2 \
  --pre-save-steps 3 \
  --post-restore-steps 3 \
  --output-dir artifacts/gen2/checkpoints/dense-reshard
```

### 10.6 Attention and CE kernel matrix

CE exact shapes:

```bash
python -m scripts.benchmark_kernels \
  --suite linear-ce \
  --batch-tokens 2048,8192 \
  --hidden-size 2048 \
  --vocab-size 32064 \
  --dtype bfloat16 \
  --implementations native_chunked,tokamax_chunked_xla,tokamax_mosaic_tpu \
  --mode forward_and_vjp \
  --method xprof_hermetic \
  --iterations 10 \
  --repeats 3 \
  --verify-native \
  --output-dir artifacts/gen2/kernels/linear-ce-v5e8
```

Run Tokamax's full-logits XLA CE only for a small control shape; it is not a memory candidate at 8192 x 32064.

Attention exact global and TP-local shapes:

```bash
python -m scripts.benchmark_kernels \
  --suite attention \
  --batch-size 1 \
  --sequence-length 2048,8192 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 64 \
  --tensor-parallel 1,2,4,8 \
  --causal \
  --dtype bfloat16 \
  --implementations jax_sdpa,laughlm_splash,tokamax_mosaic \
  --mode forward_and_vjp \
  --method xprof_hermetic \
  --iterations 10 \
  --repeats 3 \
  --verify-native \
  --output-dir artifacts/gen2/kernels/attention-gqa-v5e8
```

### 10.7 MoE/EP8 matrix

PRs 15-16 should add:

| ID | Config | Mesh | Shared weights | Expert GEMM |
|---|---|---|---|---|
| M0 | `configs/gen2/benchmarks/moe_2_73b_v5e8_ep8_replicated_shared_s2048.yaml` | `(1,1,1,8)` | replicated over EP | native ragged dot |
| M1 | `configs/gen2/benchmarks/moe_2_73b_v5e8_ep8_sharded_shared_s2048.yaml` | `(1,1,1,8)` | expert axis acts as FSDP in shared blocks | native ragged dot |
| M2 | `configs/gen2/benchmarks/moe_2_73b_v5e8_ep8_tokamax_ragged_s2048.yaml` | `(1,1,1,8)` | winner of M0/M1 | Tokamax public ragged dot |
| M3 | `configs/gen2/benchmarks/moe_2_73b_v5e8_f2_ep4_s2048.yaml` | `(1,2,1,4)` | FSDP2 | two experts/rank diagnostic |

Validate forced balanced and skewed routing before timing:

```bash
python -m scripts.validate_gen2 \
  --config configs/gen2/validation/moe_tiny_ep8.yaml \
  --forced-routing balanced,skewed,empty-expert,tied-logits \
  --compare-ep 1,2,4,8 \
  --steps 3 \
  --check-loss \
  --check-gradients \
  --check-updates \
  --check-no-drop \
  --output-dir artifacts/gen2/validation/moe-ep
```

Benchmark exact router/expert shapes. A microbatch over EP8 at sequence 2048 has 16,384 unique tokens and 32,768 Top-2 assignments, averaging 4096 assignments/expert:

```bash
python -m scripts.benchmark_kernels \
  --suite moe \
  --tokens 16384 \
  --hidden-size 2048 \
  --num-experts 8 \
  --top-k 2 \
  --expert-intermediate-size 2048 \
  --expert-parallel 8 \
  --routing-distribution balanced,skewed_2x,empty_one \
  --implementations jax_ragged_dot,tokamax_xla,tokamax_mosaic,tokamax_mosaic_tpu_v2 \
  --mode forward_and_vjp \
  --method xprof_hermetic \
  --iterations 10 \
  --repeats 3 \
  --verify-native \
  --output-dir artifacts/gen2/kernels/moe-v5e8
```

Full-step commands:

```bash
for cfg in configs/gen2/benchmarks/moe_2_73b_v5e8_*_s2048.yaml; do
  name=$(basename "${cfg}" .yaml)
  for trial in 1 2 3; do
    python -m scripts.benchmark_gen2 \
      --config "${cfg}" \
      --synthetic-data \
      --seed 20260904 \
      --warmup-steps 10 \
      --xprof-steps 5 \
      --measure-steps 50 \
      --output-dir "artifacts/gen2/moe/${name}/trial-${trial}"
  done
done
```

### 10.8 GDN2/Splash matrix

```bash
python -m scripts.validate_gen2 \
  --config configs/gen2/validation/gdn2_tiny.yaml \
  --sequence-length 32,64,128 \
  --compare-implementations token_reference,native_chunked \
  --check-final-state \
  --check-gradients \
  --check-packed-resets \
  --output-dir artifacts/gen2/validation/gdn2

python -m scripts.benchmark_kernels \
  --suite gdn2 \
  --batch-size 1 \
  --sequence-length 2048,8192 \
  --hidden-size 2048 \
  --key-heads 16 \
  --value-heads 32 \
  --key-head-dim 128 \
  --value-head-dim 128 \
  --conv-kernel-size 4 \
  --chunk-size 64 \
  --dtype bfloat16 \
  --implementations native_chunked \
  --mode forward_and_vjp \
  --method xprof_hermetic \
  --iterations 10 \
  --repeats 3 \
  --output-dir artifacts/gen2/kernels/gdn2-v5e8

python -m scripts.benchmark_gen2 \
  --config configs/gen2/benchmarks/gdn2_splash_3to1_1_34b_v5e8_d2_f2_t2_s2048.yaml \
  --synthetic-data \
  --seed 20260904 \
  --warmup-steps 10 \
  --xprof-steps 5 \
  --measure-steps 50 \
  --output-dir artifacts/gen2/gdn2/native-d2-f2-t2-trial-1
```

The private Tokamax experiment is a separate process and verifies the exact source revision before import:

```bash
python -m scripts.experiments.benchmark_tokamax_private \
  --required-tokamax-revision 609307ca5a98ccd831bb534b8d283281750c17df \
  --op kda \
  --config configs/gen2/experiments/kda_laughlm_shape_v5e8.yaml \
  --verify-against native_chunked \
  --method xprof_hermetic \
  --output-dir artifacts/gen2/experiments/kda-v5e8
```

Failure of semantic parity ends the KDA experiment before performance timing.

### 10.9 HLO and XProf inspection

For a selected config:

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/laughlm-gen2-hlo --xla_dump_hlo_as_text"
python -m scripts.compile_gen2 \
  --config configs/gen2/benchmarks/dense_1_33b_v5e8_d2_f2_t2_s2048.yaml \
  --output-dir artifacts/gen2/compile/dense-d2-f2-t2-hlo
python -m scripts.inspect_collectives \
  --hlo-dir /tmp/laughlm-gen2-hlo \
  --expect all-gather,reduce-scatter,all-reduce \
  --reject-all-reduce-inside gradient_accumulation_scan \
  --output artifacts/gen2/compile/dense-d2-f2-t2-hlo/collectives.json
```

For every promoted configuration, review:

- step-time breakdown and TensorCore utilization;
- HBM peak and buffer timeline;
- all-gather, reduce-scatter, all-reduce, and all-to-all time/bytes;
- collective overlap with compute;
- input and output layout conversions;
- compile time, executable size, and persistent cache reuse;
- host/device idle time;
- kernel implementation names and tuning keys.

### 10.10 Promotion thresholds

Correctness is mandatory. Performance is judged over three independent trials after compilation:

| Gate | Threshold |
|---|---|
| FP32 tiny loss/output parity | `rtol <= 1e-5`, `atol <= 1e-6` |
| BF16 TPU loss parity | `rtol <= 3e-3`, `atol <= 3e-3` |
| Gradient direction | cosine similarity `>= 0.999` on representative leaves |
| One-update relative L2 difference | `<= 5e-3` in BF16 qualification |
| NaN/Inf | zero |
| Sample coverage | every expected ID exactly once across unique replicas |
| Checkpoint | step/tokens/RNG/data cursor exact; next three losses within dtype tolerance |
| Trial stability | throughput coefficient of variation `<= 3%` |
| Kernel promotion | at least 5% full-step throughput gain, or at least 10% peak-HBM reduction with no more than 2% throughput regression |
| Runtime promotion | no unexplained collectives and no persistent full state replicas |
| MoE | zero dropped tokens; router max/mean and entropy recorded; no dead expert in smoke run |

A kernel that only wins a microbenchmark but loses the full train step is not promoted.

## 11. Dependency and runtime modernization

### Gap audit

| Component | LaughLM now | Current reference baseline (2026-09-04) | Gap |
|---|---:|---:|---|
| Python | `>=3.10` | 3.12 | Tokamax and current MaxText require 3.12+. |
| JAX | 0.4.38 | 0.11.1 | Major API/runtime gap; explicit sharding and current Tokamax require modernization. |
| JAXLIB | 0.4.38 | 0.11.1 | Must match JAX exactly. |
| Flax | 0.10.2 | 0.12.9 | Linen compatibility must be proven; do not migrate to NNX concurrently. |
| Optax | 0.2.4 | 0.2.8 | State-tree behavior and aliases need regression tests. |
| Orbax checkpoint | 0.11.4 | 0.12.4 | Move toward current public/V1 APIs and abstract sharded restore. |
| Tokamax | optional, unpinned | 0.0.13 | Pin exact version and public capabilities. |
| XProf | not pinned; current integration disabled | 2.23.1 | Must match profiler plugin/runtime. |
| Grain | optional, unpinned | 0.2.18 | Pin in Gen-2 data extra; not required for synthetic runtime benchmarks. |
| Pydantic | `>=2.0` | 2.13.5 | Pin to make schema behavior reproducible. |
| libtpu | inherited from image/environment | 0.0.47 candidate; MaxText requires at least 0.0.46 | Pin the validated wheel/container combination. |

The current release pages list JAX 0.11.1, Flax 0.12.9, Optax 0.2.8, Orbax 0.12.4, Tokamax 0.0.13, XProf 2.23.1, Grain 0.2.18, and Pydantic 2.13.5. MaxText's generated requirements use the same JAX/Flax/Optax/Orbax/Tokamax generation, mostly as lower bounds.

One upstream warning matters: MaxText's TPU overrides currently cap XProf at 2.23.0, while Tokamax 0.0.13 requires XProf 2.23.1 or newer. LaughLM does **not** install MaxText, so it should not copy that override. It must resolve and qualify its own coherent lock.

### Candidate Gen-2 direct pins

Use this only as the lock input, not as proof of production compatibility:

```text
Python                    3.12
jax                       0.11.1
jaxlib                    0.11.1
flax                      0.12.9
optax                     0.2.8
orbax-checkpoint          0.12.4
tokamax                   0.0.13
xprof                     2.23.1
grain                     0.2.18
pydantic                  2.13.5
libtpu                    0.0.47
```

Pallas/Mosaic TPU APIs ship with the pinned JAX/JAXLIB/libtpu stack; do not add a fictitious independent `pallas` dependency.

### Reproducible production baseline

PR 1 should create:

```text
requirements/legacy.in
requirements/gen2-cpu.in
requirements/gen2-tpu-v5e.in
requirements/locks/legacy-py312.txt
requirements/locks/gen2-cpu-py312.txt
requirements/locks/gen2-tpu-v5e-py312.txt
containers/gen2-tpu/Dockerfile
```

Requirements for the lock process:

1. Pin every direct and transitive package with hashes.
2. Record the resolver and resolver version in the generated header.
3. Pin the base image by digest, not a floating tag.
4. Record the libtpu wheel hash and `jax.print_environment_info()` output.
5. Run `pip check`, full CPU tests, config parsing, checkpoint restore, and a v5e smoke before blessing a lock.
6. Never use `pip install --upgrade "jax[tpu]"` in a production runbook without the lock/constraint file.
7. Do not install MaxText as a runtime dependency.
8. Keep Tokamax in a `gen2-kernels` extra until at least one public kernel is promoted; native JAX must work without it.

Migration gates:

1. Current lock -> modern CPU lock: all existing tests and serialized-config tests.
2. Current TPU lock -> modern TPU lock: PMAP and FSDP tiny parity.
3. Restore a checkpoint written by Orbax 0.11.4 under 0.12.4 and continue three deterministic steps.
4. Produce a fresh checkpoint under 0.12.4 and verify same-layout and changed-layout restore.
5. Verify XProf start/stop succeeds; no broad catch may turn a requested trace into a silent no-op.

Only after these gates should the modern lock be called the production baseline.

## 12. Risk register and exit criteria

| Risk | Failure signal | Mitigation / exit criterion |
|---|---|---|
| Duplicate FSDP samples | Same sample IDs on multiple FSDP ranks; inflated tokens/s | Unique-ID test across `data x fsdp`; corrected metric baseline. |
| Axis collision | One physical axis appears twice in a tensor spec; compiler error or unintended collective | Static v2 rule resolver rejects duplicate physical axes per `PartitionSpec`. |
| FSDP is only nominal | Full params/moments persist on every device | Sharding-tree assertion, HBM scaling, and HLO all-gather/reduce-scatter evidence. |
| TP communication dominates v5e | Lower throughput as TP grows; long reduce/all-gather regions | Compare D2-D5; choose smallest TP that meets HBM. No requirement to use TP8. |
| GQA KV expansion erases savings | KV buffers or traffic scale from 8 to 32 heads globally | Local-head inspection and XProf; direct GQA candidate benchmark. |
| Incorrect distributed CE | Loss/z-loss differs by TP factor or ignored labels | Dense-vs-distributed oracle over TP1/2/4/8 and adversarial labels. |
| Optimizer replication | Adam moments appear as `P()` | Structural startup assertion and checkpoint metadata digest. |
| Reduction occurs per microbatch | All-reduce appears inside GA scan | HLO rejection gate and post-scan layout transition. |
| Orbax API/format drift | Restore fails or chooses stale/incomplete checkpoint | Versioned metadata, committed-step marker, abstract target restore, fault injection. |
| Kernel API drift | Tokamax upgrade changes signatures or selected backend | Exact pin, top-level imports only, startup capability matrix, explicit implementation. |
| Autotuning nondeterminism | Different tiles/numerics across runs | Offline tune, serialize result, hash it, and load only a qualified key. |
| Private Tokamax dependency leaks | `_src` import in package or production config | CI static check forbids `tokamax._src` outside `scripts/experiments`. |
| Pallas blocks compiler overlap | Kernel wins alone but full step regresses | Full-step XProf is the promotion authority. |
| MoE imbalance | High max/mean assignments, dead experts, all-to-all tail | Router metrics, skew tests, auxiliary loss, no-drop ragged baseline. |
| EP role confusion | Expert all-to-all becomes shared-weight gather/RS | Separate shared and dispatched activation logical axes; inspect HLO collectives. |
| GDN state leaks across packed docs | Packed loss differs from independently processed documents | Segment-reset golden test for output, final state, and VJP. |
| GDN BF16 instability | NaNs or trajectory drift at long sequence | FP32 recurrence accumulators/gates where required; sequence 2K then 8K gates. |
| Model-size drift | Config no longer matches 1.33B/2.73B/0.92B claims | Exact parameter-count unit tests and manifest fields. |
| Dependency modernization regresses stable trainer | PMAP/FSDP loss or checkpoint mismatch | Dual-lock CI and three-step continuation equivalence before default change. |

### Branch-level release exits

Dense 1.33B exits candidate status only when:

- corrected FSDP and one DP/FSDP/TP mesh pass all parity tests;
- native distributed CE never creates global logits;
- same-layout and cross-layout resume pass;
- v5e HBM leaves an operational margin of at least 10%;
- three-trial throughput and XProf reports are checked in;
- PMAP/current FSDP regression tests remain green.

MoE 2.73B exits experimental status only when:

- exact total/active counts pass;
- forced-routing EP1 and EP8 match;
- no tokens are dropped;
- load-balance metrics remain finite and every expert receives traffic in the smoke run;
- EP8 save/resume preserves router, expert, optimizer, RNG, and data position;
- native JAX is a supported production path even if Tokamax is unavailable.

GDN2/Splash exits experimental status only when:

- token recurrence and chunked recurrence match in output, final state, and VJP;
- packed-document state resets pass;
- the exact 3:1 schedule and parameter count are in metadata;
- sequence 2048 and 8192 complete without numerical instability;
- checkpoint continuation crosses a GDN/Splash cycle boundary correctly;
- no private Tokamax API is needed.

## 13. Final recommendation

The shortest safe path is not “port MaxText” and not “turn on Tokamax.” It is:

1. modernize and lock the JAX/TPU environment;
2. make LaughLM's existing global-array FSDP mathematically correct and observable;
3. add a compact DP/FSDP/TP runtime with native JAX references;
4. qualify Dense 1.33B;
5. branch independently into MoE/EP and GDN2/Splash;
6. let exact v5e-8 evidence decide each Tokamax promotion.

That preserves LaughLM's strongest existing properties—small understandable code, deterministic native fallbacks, and strict checkpoint metadata—while borrowing the distributed layout lessons that MaxText has already learned and using Tokamax for what it is best positioned to provide: replaceable, measured kernels rather than trainer semantics.

## Primary upstream references

- [Tokamax repository and development status](https://github.com/openxla/tokamax)
- [Tokamax public top-level exports](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/tokamax/__init__.py)
- [Tokamax linear softmax cross-entropy API](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py)
- [Tokamax attention API](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/tokamax/_src/ops/attention/api.py)
- [Tokamax ragged-dot API](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/tokamax/_src/ops/ragged_dot/api.py)
- [Tokamax benchmarking guide](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/docs/benchmarking.md)
- [Tokamax autotuning guide](https://github.com/openxla/tokamax/blob/609307ca5a98ccd831bb534b8d283281750c17df/docs/autotuning.md)
- [Tokamax experimental operations tree](https://github.com/openxla/tokamax/tree/609307ca5a98ccd831bb534b8d283281750c17df/tokamax/_src/ops/experimental)
- [MaxText base mesh and logical-axis configuration](https://github.com/AI-Hypercomputer/maxtext/blob/501f3b828793adc321dda02f524f014324395100/src/maxtext/configs/base.yml)
- [MaxText mesh construction](https://github.com/AI-Hypercomputer/maxtext/blob/501f3b828793adc321dda02f524f014324395100/src/maxtext/utils/maxtext_utils.py)
- [MaxText gradient accumulation](https://github.com/AI-Hypercomputer/maxtext/blob/501f3b828793adc321dda02f524f014324395100/src/maxtext/utils/gradient_accumulation.py)
- [MaxText MoE implementation](https://github.com/AI-Hypercomputer/maxtext/blob/501f3b828793adc321dda02f524f014324395100/src/maxtext/layers/moe.py)
- [MaxText Qwen3 Gated Delta implementation](https://github.com/AI-Hypercomputer/maxtext/blob/501f3b828793adc321dda02f524f014324395100/src/maxtext/models/qwen3.py)
- [JAX explicit/automatic/manual parallelism](https://docs.jax.dev/en/latest/parallel.html)
- [JAX ragged dot](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html)
- [JAX ragged all-to-all](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html)
- [JAX rematerialization](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [JAX/Pallas distributed TPU programming](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
- [Orbax sharded PyTree checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html)
- [Orbax API migration index](https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html)
- [Flax multi-device sharding](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html)
- [XProf profile capture](https://openxla.org/xprof/capturing_profiles)
