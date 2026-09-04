# 3D Parallelism Implementation Roadmap

Status: proposed  
Target: `LaughLM` JAX/Flax trainer on TPU  
Scope: data parallelism + fully sharded data parallelism + tensor parallelism, with an optional sequence-parallel activation layout

## Objective

Extend the current FSDP training path into a topology-aware SPMD trainer that can train models too large for the existing mesh while using all TPU cores efficiently.

The target design follows MaxText-style JAX sharding:

- `data`: replicas that process unique batch shards.
- `fsdp`: shards persistent parameters, gradients, and optimizer state; it also participates in activation-batch sharding.
- `tensor`: shards model dimensions and cooperatively executes each layer.
- Sequence parallelism initially reuses the `tensor` group as an alternate activation layout. It is not a fourth physical mesh axis in the first release.

The JAX global-array and `PartitionSpec` model provides ZeRO-3-like parameter and optimizer-state sharding. This implementation should not add manual parameter all-gather wrappers unless compiler inspection demonstrates that explicit collectives are required.

## Non-negotiable invariants

1. A training example is unique across `data * fsdp`; tensor ranks operate on the same examples.
2. Global batch size is:

   ```text
   micro_batch_per_replica * data_axis_size * fsdp_axis_size
   ```

   The tensor axis must not multiply the global batch size.
3. Parameter and optimizer-state shardings are structurally identical.
4. Every tensor-parallel collective is visible in lowered HLO and justified by a layer boundary.
5. No logical tensor dimension may map to the same physical mesh axis twice in one `PartitionSpec`.
6. Single-device, pure-FSDP, and 3D paths must agree numerically within the selected dtype tolerances.
7. Existing `runtime.backend: fsdp` configurations remain supported while the 3D backend is introduced.
8. Checkpoints must fail loudly on incompatible mesh metadata until cross-mesh resharding is deliberately implemented.

## Proposed architecture

Keep `fsdp_train_step.py` as the shared jitted step because its explicit input, state, and output shardings already fit the target model. Refactor trainer orchestration into a shared internal SPMD base and expose two stable routes:

- `FSDPTrainer`: existing pure-FSDP behavior and configuration compatibility.
- `Parallel3DTrainer`: data + FSDP + tensor behavior, registered as `runtime.backend: parallel3d`.

Do not silently reinterpret an existing FSDP configuration as 3D. The first 3D release should use a distinct backend and explicit mesh configuration.

### Logical-axis model

The current `embed` logical name is used for both weights and activations. That is insufficient because an FSDP weight dimension and a tensor-parallel activation dimension may need different physical mappings. Split the logical vocabulary before adding tensor-parallel layers.

Suggested names:

| Category | Logical axes |
| --- | --- |
| Parameters | `weight_embed`, `weight_mlp`, `weight_heads`, `weight_kv_heads`, `vocab`, `norm`, `layers` |
| Activations | `activation_batch`, `activation_sequence`, `activation_embed`, `activation_mlp`, `activation_heads`, `activation_kv_heads` |

Logical-axis rules must accept `string | list[string] | null`. A list maps one logical dimension over multiple physical axes, for example:

```yaml
logical_axis_rules:
  activation_batch: [data, fsdp]
  weight_embed: fsdp
  weight_mlp: tensor
  activation_embed: tensor
  activation_heads: tensor
```

Validation must reject unknown axes, repeated physical axes within one resulting spec, and incompatible model dimensions before model initialization.

### Initial partitioning contract

| Tensor | Shape | Initial regular-TP sharding |
| --- | --- | --- |
| Token input | `[GA, B, T]` | `P(None, ("data", "fsdp"), None)` |
| Residual activation | `[B, T, D]` | `P(("data", "fsdp"), None, "tensor")` |
| Query | `[B, T, H, Dh]` | `P(("data", "fsdp"), None, "tensor", None)` |
| Key/value | `[B, T, Hkv, Dh]` | Same as query when `Hkv % TP == 0`; otherwise explicitly replicated |
| MLP gate/up weight | `[D, M]` | `P("fsdp", "tensor")` |
| MLP down weight | `[M, D]` | `P("tensor", "fsdp")` |
| Embedding | `[V, D]` | Candidate `P("tensor", "fsdp")`; validate tied-logit behavior before freezing |

`GA` denotes gradient-accumulation steps. Exact optimizer-state specs derive from the parameter tree rather than a separately maintained ruleset.

## Milestones

| Milestone | Result | TPU required |
| --- | --- | --- |
| M0 | Baseline and contracts frozen | Yes, for baseline metrics |
| M1 | Correct FSDP batch semantics | Yes |
| M2 | Split logical axes and validation | No for most tests |
| M3 | Real 3D mesh and backend routing | Yes |
| M4 | Tensor-parallel MLP | Yes |
| M5 | Tensor-parallel attention | Yes |
| M6 | Sequence-parallel activation mode | Yes |
| M7 | Multi-host input and topology support | Multi-host TPU |
| M8 | Checkpoint, restore, and export | Yes |
| M9 | Performance qualification and release | Target TPU topology |

## M0 - Freeze the baseline and contracts

Goal: make correctness and performance regressions measurable before changing semantics.

Work:

- Record the current 1.3B FSDP benchmark configuration, compile time, step time, unique tokens/second, MFU, peak HBM, and checkpoint duration.
- Preserve the existing observed baseline of approximately 41.9k reported tokens/second and 42.6% MFU, but relabel it as pre-correction because current batch semantics can duplicate examples across FSDP ranks.
- Add a tiny deterministic model/config fixture with dropout disabled.
- Define dtype-specific parity tolerances for logits, loss, gradients, and one optimizer update.
- Capture parameter, optimizer-state, input, and output sharding trees as golden structural assertions.

Exit criteria:

- One reproducible pure-FSDP benchmark report exists.
- A one-step single-device reference test is deterministic.
- Metrics distinguish reported tokens from unique training tokens.

Rollback rule: no runtime behavior changes belong in this milestone.

## M1 - Correct FSDP batch semantics first

Goal: eliminate duplicated examples and compute before introducing tensor parallelism.

Primary files:

- `training/fsdp_trainer.py`
- `utils/sharding_factory.py`
- `config/validation.py`
- current FSDP YAML configurations
- FSDP trainer and sharding tests

Work:

- Map the activation batch dimension over `("data", "fsdp")`, not only `data`.
- Calculate global batch size with `data_size * fsdp_size` unique batch replicas.
- Keep gradient-accumulation semantics unchanged and document whether configured micro-batch size is per unique replica or per host.
- Correct token accounting and throughput reporting to count unique tokens.
- Move long-running token counters away from device `int32`; prefer a host-side `int64` counter or an explicitly safe scalar policy.
- Remove any factory behavior that silently rewrites requested axis sizes.

Tests:

- Partition-spec assertion for input batch.
- Unique sample-ID coverage across `data * fsdp`, with no duplicates.
- Loss and update parity between a tiny single-device run and corrected FSDP.
- Global-batch and tokens-processed accounting tests.

TPU gate:

- Corrected FSDP consumes unique examples on every FSDP rank.
- No unexpected full parameter or optimizer replication appears in memory analysis.
- A fresh corrected-FSDP performance baseline is recorded.

Exit criteria: corrected FSDP is merged and stable before M2-M5 are enabled.

Rollback rule: retain a configuration migration note, but do not preserve the duplicate-batch behavior as a compatibility mode.

## M2 - Refactor logical axes and validation

Goal: make weight and activation mappings independently expressible.

Primary files:

- logical-axis config/schema definitions
- `config/validation.py`
- model parameter annotations
- activation constraint helpers
- `utils/sharding_factory.py`

Work:

- Replace overloaded names such as `embed`, `mlp`, and `heads` with the parameter/activation names in the logical-axis model above.
- Extend rule values from `Optional[str]` to `str | list[str] | null`.
- Normalize YAML lists to physical-axis tuples when producing `PartitionSpec` values.
- Validate axis existence, axis-size products, divisibility, and repeated-axis conflicts.
- Add a compatibility translator for old pure-FSDP config keys, with deprecation warnings and deterministic output.
- Add explicit sharding constraints at important activation boundaries rather than relying only on parameter annotations.

Tests:

- Schema round-trip for scalar, tuple/list, and replicated mappings.
- Rejection tests for unknown or duplicate physical axes.
- Golden parameter and activation spec trees for FSDP and 3D candidates.
- Legacy FSDP configuration translation test.

Exit criteria: no model tensor depends on an ambiguous `embed`/`mlp`/`heads` mapping.

Rollback rule: the compatibility translator allows the old configs to remain usable while model internals adopt the new names.

## M3 - Build the real 3D mesh and backend

Goal: create an explicit, validated `data x fsdp x tensor` execution route.

Primary files:

- `distributed/mesh.py`
- `utils/sharding_factory.py`
- `config/validation.py`
- `scripts/train.py`
- new `training/parallel3d_trainer.py` or shared `training/spmd_trainer.py`
- 3D example configurations

Work:

- Replace the nominal `maxtext_3d` branch with a mesh that includes every configured active axis and never silently changes sizes.
- Require `data * fsdp * tensor == global_device_count` for the MVP. Sequence and pipeline axes must be one.
- Register `parallel3d` in the trainer factory/CLI.
- Share state creation, compilation, metrics, and checkpoint plumbing with `FSDPTrainer`.
- Preserve physical axis order `data, fsdp, tensor, sequence, pipeline` where axes are present.
- Log the resolved mesh, device coordinates, process ownership, and logical rules once at startup.

Candidate v5e-8 meshes:

```yaml
# Memory-first
data: 1
fsdp: 4
tensor: 2

# Batch/throughput balance
data: 2
fsdp: 2
tensor: 2
```

Tests:

- Device-count and invalid-product validation.
- Mesh-axis ordering and `NamedSharding` construction.
- Backend registry and configuration parsing.
- State initialization without accidental full replication.

TPU gate:

- Both candidate meshes compile a tiny model.
- Device visualization confirms expected shard placement.
- State memory decreases in accordance with FSDP and TP partitioning.

Exit criteria: the 3D backend can initialize, checkpoint metadata can describe it, and a no-op/tiny step runs on all devices.

Rollback rule: `fsdp` remains the production backend until M5 full-step parity passes.

## M4 - Tensor-parallelize the MLP

Goal: establish the first end-to-end tensor-parallel layer with predictable collectives.

Primary files:

- transformer MLP module(s)
- model axis annotations
- activation constraint helpers
- parity and HLO inspection tests

Work:

- Shard gate/up projections as `P("fsdp", "tensor")`.
- Shard the intermediate activation over `tensor`.
- Shard down projection as `P("tensor", "fsdp")`.
- Constrain residual output to the selected activation layout.
- Verify that bias, normalization, dropout RNG, and residual operations do not introduce unintended replication.
- Inspect lowered HLO for the expected reduction at the down-projection boundary and for absence of redundant all-gathers.

Tests:

- MLP forward parity.
- Input-gradient and parameter-gradient parity.
- One optimizer-update parity.
- Partition and collective structural assertions.
- Invalid `intermediate_size % tensor_size` rejection.

TPU gate:

- MLP TP uses all tensor ranks.
- Peak HBM and step-time effects are measured against corrected FSDP.
- Numerical error remains within the M0 tolerance.

Exit criteria: MLP TP passes forward/backward/update parity and HLO review.

Rollback rule: keep an explicit `tensor_parallel_mlp: false` development flag until full-model M5 passes; remove or hide it from production configs afterward.

## M5 - Tensor-parallelize attention

Goal: complete transformer-block tensor parallelism.

Primary files:

- attention projection modules
- attention kernel selection and Splash Attention shard-map code
- model validation
- attention and full-step parity tests

Work:

- Disable fused QKV for the first correctness gate.
- Require `num_attention_heads % tensor_size == 0`.
- Require `num_kv_heads % tensor_size == 0`, or support an explicit `replicate_kv_heads` policy. Never choose replication silently.
- Shard Q/K/V head dimensions over `tensor`; keep head dimension local.
- Execute attention locally per tensor rank and reduce/repartition at the output projection boundary.
- Replace the hard-coded Splash Attention spec `P("data", None, None, None)` with a spec derived from the resolved activation layout.
- Use XLA SDPA for initial parity. Restore Splash Attention only after its shard-map and mask layouts pass the same tests.
- Treat tied embedding/output logits as a separate validation case because vocabulary and hidden sharding interact.
- Reintroduce fused QKV only with a structured layout that preserves separate Q and KV head groups; do not shard the flat concatenated output dimension blindly.

Tests:

- MHA forward/backward parity.
- GQA parity with divisible KV heads.
- Explicit replicated-KV parity if that policy is implemented.
- Mask and causal-attention parity.
- Full transformer-block and one-train-step parity.
- HLO collective inspection.

TPU gate:

- Full-model training step runs with 3D sharding on all cores.
- No full-size Q/K/V or MLP intermediates are unexpectedly replicated.
- Loss trajectory matches corrected FSDP over a short deterministic run.

Exit criteria: `parallel3d` becomes functionally complete for the supported model family.

Rollback rule: keep Splash Attention and fused QKV disabled for 3D if either cannot satisfy parity and sharding checks; XLA SDPA plus separate projections is an acceptable initial release.

## M6 - Add sequence-parallel activation mode

Goal: reduce residual/norm activation memory without introducing another physical axis.

Configuration shape:

```yaml
tp_activation_layout: feature  # feature | sequence
```

Work:

- `feature`: residual `[B, T, D] -> P(("data", "fsdp"), None, "tensor")`.
- `sequence`: residual `[B, T, D] -> P(("data", "fsdp"), "tensor", None)`.
- Keep tensor-parallel MLP intermediates and attention head shards on the tensor group.
- Add explicit transitions between sequence-sharded residuals and head/feature-sharded compute.
- Validate normalization reductions and dropout RNG semantics under sequence sharding.
- Measure whether transition collectives erase the activation-memory gain.

Tests:

- Feature-layout versus sequence-layout forward/backward parity.
- Norm statistics parity.
- Padding/mask behavior when sequence length is not divisible; initially reject unsupported lengths.
- Collective-count and peak-activation-memory comparison.

TPU gate: sequence mode must reduce peak activation HBM for a long-context configuration without an unacceptable step-time regression.

Exit criteria: sequence mode is opt-in and independently benchmarked.

Rollback rule: feature mode remains the default.

## M7 - Make input and mesh construction multi-host safe

Goal: support pod and multi-slice execution without assuming every host owns a global batch.

Primary files:

- dataset/input pipeline integration
- SPMD trainer batch conversion
- `distributed/mesh.py`
- distributed startup and diagnostics

Work:

- Have each process produce process-local data only.
- Build global arrays with `jax.make_array_from_process_local_data(input_sharding, local_data, global_shape=...)`.
- Validate local shape, global shape, process count, and addressable-device ownership before compilation.
- Add sample-ID diagnostics to prove global uniqueness across hosts.
- Preserve ICI/DCN hierarchy for multi-slice meshes: prefer FSDP and tensor traffic on ICI, and map data replication across DCN when topology permits.
- Stop flattening ICI/DCN dimensions into a topology-oblivious size before mesh creation.
- Fail with a topology report when the requested mesh cannot be laid out contiguously.

Tests:

- Mocked process-local/global shape calculations.
- Multi-process sample uniqueness.
- Hierarchical mesh resolution tests.
- Startup failure tests for inconsistent host batches.

Multi-host TPU gate:

- At least one two-host run completes initialization, training, save, and restore.
- Cross-host unique-example accounting is correct.
- Collective traces match intended ICI/DCN placement.

Exit criteria: the same config can scale from one host to its supported larger topology by changing mesh sizes and batch parameters only.

Rollback rule: reject unsupported multi-slice layouts instead of falling back to a flattened mesh.

## M8 - Checkpoint, restore, and export

Goal: make 3D training operationally recoverable.

Primary files:

- checkpoint manager and metadata validation
- trainer restore path
- Hugging Face/export utilities
- checkpoint integration tests

Work:

- Store physical mesh axes/sizes, logical-axis rules, parameter specs, dtype, model config, and optimizer structure.
- Validate same-layout save/restore first.
- Add an explicit canonical gather/unshard path for export and single-device evaluation.
- Keep cross-mesh restore unsupported until a deliberate resharding path is implemented and tested.
- Add actionable errors for legacy checkpoints with ambiguous logical axes.
- Verify that metrics and host-side token counters resume correctly.

Tests:

- Same-layout 3D save/restore equality.
- Continued-training loss parity after restore.
- Incompatible-layout rejection.
- Canonical gathered parameter equality and HF export smoke test.

TPU gate: interrupt and resume a short 3D job with no loss discontinuity beyond deterministic tolerance.

Exit criteria: a production run can recover from its latest checkpoint and export canonical weights.

Rollback rule: checkpoint metadata versioning must allow the code to identify, not guess, old layouts.

## M9 - Performance qualification and release

Goal: prove that 3D improves feasible model size or useful throughput on the target TPU topology.

Compare every candidate against the corrected M1 FSDP baseline using:

- unique tokens/second, not replicated tokens;
- model FLOPs utilization;
- peak HBM and memory headroom;
- compile time and executable size;
- step-time distribution after warmup;
- collective time/share and bytes by collective type;
- input stall percentage;
- checkpoint save/restore overhead;
- loss and gradient-norm trajectories.

Qualification matrix:

| Dimension | Required comparisons |
| --- | --- |
| Mesh | pure FSDP, `1x4x2`, `2x2x2` on v5e-8 where model size permits |
| Attention | XLA SDPA; Splash only if M5-qualified |
| Activation layout | feature; sequence for long context |
| Model size | one parity-sized model and one model that cannot fit the corrected FSDP baseline |
| Context | current production length and one long-context stress case |

Release gates:

- No correctness failure in deterministic short runs.
- 3D enables a larger model/context or demonstrates a documented useful performance/memory tradeoff.
- No unexplained collective or full-tensor replication remains in HLO/profile review.
- Checkpoint recovery and canonical export pass.
- Pure FSDP regression suite remains green.

## Configuration sketch

This is a target shape, not a drop-in config until M2-M5 are implemented:

```yaml
runtime:
  backend: parallel3d

mesh:
  ici_data_parallelism: 1
  ici_fsdp_parallelism: 4
  ici_tensor_parallelism: 2
  ici_sequence_parallelism: 1
  ici_pipeline_parallelism: 1

optimizations:
  sharding_strategy: parallel3d
  tp_activation_layout: feature
  fused_qkv: false

logical_axis_rules:
  activation_batch: [data, fsdp]
  activation_sequence: null
  activation_embed: tensor
  activation_heads: tensor
  activation_kv_heads: tensor
  weight_embed: fsdp
  weight_mlp: tensor
  weight_heads: tensor
  weight_kv_heads: tensor
  vocab: tensor
  norm: null
  layers: null
```

The final embedding/logit rules must be frozen only after tied and untied output-head tests pass.

## Test inventory to add

1. `test_fsdp_unique_batch_semantics`
2. `test_logical_axis_rule_tuple_mapping`
3. `test_partition_spec_rejects_duplicate_physical_axis`
4. `test_parallel3d_mesh_product_and_axis_order`
5. `test_mlp_tp_forward_backward_update_parity`
6. `test_attention_tp_mha_parity`
7. `test_attention_tp_gqa_parity`
8. `test_parallel3d_full_step_parity`
9. `test_sequence_parallel_norm_and_mask_parity`
10. `test_process_local_batch_to_global_array`
11. `test_parallel3d_checkpoint_same_layout_restore`
12. `test_parallel3d_checkpoint_incompatible_layout_rejected`
13. `test_canonical_gather_and_export`
14. TPU-only HLO, HBM, and collective-profile qualification tests

CPU tests should cover schemas, shape math, tree mappings, validation, and small numerical parity where JAX supports the operation. TPU tests are mandatory for mesh placement, compiler collectives, memory behavior, and real performance.

## Risk register

| Risk | Mitigation |
| --- | --- |
| Duplicate examples across FSDP ranks | Complete M1 and sample-ID checks before TP work |
| Ambiguous logical axes | Split parameter and activation names in M2 |
| Invalid GQA/KV partitioning | Divisibility validation or explicit KV replication policy |
| Splash shard-map mismatch | Qualify XLA SDPA first; derive shard maps from resolved layout |
| Optimizer-state replication | Derive state sharding from parameter tree and assert it structurally |
| Incorrect norm reductions | Dedicated sequence/feature layout parity tests |
| Tied embedding/logit conflict | Separate tied/untied tests before freezing vocab layout |
| Host-local/global batch mismatch | Use process-local-to-global JAX array construction |
| Checkpoint incompatibility | Version metadata and reject unsupported reshards |
| Token-counter overflow | Host `int64` accounting before long runs |
| Poor inter-slice placement | Preserve ICI/DCN topology and validate resolved device mesh |
| Excess TP communication | Inspect HLO/profile at every layer milestone, not only at release |

## Explicitly out of scope for the first release

- Pipeline parallelism.
- A separate physical context/sequence mesh axis.
- MoE/expert parallelism.
- Automatic arbitrary-mesh checkpoint resharding.
- Fused QKV on the first tensor-parallel correctness path.
- Guaranteed Splash Attention support before shard-map parity is proven.
- Silent mesh-size correction or silent KV-head replication.

## Recommended implementation order

```text
M0 baseline
  -> M1 correct FSDP batch semantics
  -> M2 logical-axis refactor
  -> M3 3D mesh/backend
  -> M4 TP MLP
  -> M5 TP attention
  -> M6 sequence mode
  -> M7 multi-host/topology
  -> M8 checkpoint/export
  -> M9 performance release
```

M7 input-array work may begin after M3, and M8 metadata work may begin after M2, but neither should be declared complete before full-model M5 parity.

## Definition of done

The feature is complete when a `parallel3d` configuration:

- uses every requested TPU core with a validated `data x fsdp x tensor` mesh;
- processes unique examples across `data x fsdp`;
- shards parameters and optimizer state with FSDP and layer compute with TP;
- passes forward, backward, update, short-run, checkpoint, and export parity gates;
- supports process-local input on the target multi-host topology;
- has measured HBM, throughput, MFU, compilation, and collective behavior;
- either trains a model/context that corrected FSDP cannot fit or provides a clearly documented useful tradeoff;
- leaves the existing pure-FSDP backend correct and supported.
