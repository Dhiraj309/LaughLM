# LaughLM Gen-2 commit-by-commit implementation roadmap

Status: execution tracker  
Created: 2026-09-04  
Implementation stories: 110 commits across 19 pull requests  
Heavy TPU work: manual only  

This is the executable companion to the
[Gen-2 implementation plan](GEN2_IMPLEMENTATION_PLAN.md). The implementation
plan owns architecture decisions, upstream research, model dimensions,
benchmark shapes, and promotion thresholds. This roadmap turns those decisions
into small stories, where every story produces exactly one reviewable commit.

The current PMAP and FSDP behavior remains the default throughout this roadmap.
All Gen-2 runtime, Dense 1.33B, MoE/EP, GDN2, and Tokamax behavior is opt-in.

## 1. How to use this tracker

### Status flags

| Flag | Meaning | When to use it |
|---|---|---|
| 🟢 | Ready | All dependencies are complete; this is an allowed next story. |
| ⬜ | Planned | Not started or waiting for an earlier story. |
| 🟡 | In progress | Code is actively being developed but the story commit is not complete. |
| 🟣 | TPU validation pending | The implementation commit exists and CPU CI passes, but its named manual TPU evidence is outstanding. |
| ✅ | Done | The commit exists and every acceptance item assigned to that story passes. |
| ⛔ | Blocked | Progress requires an external decision, missing capability, or failed prerequisite. |
| ↩️ | Rolled back | The commit was reverted; record the revert SHA and reason in the decision log. |
| ❌ | No-go | A benchmark or experiment failed its promotion gate and will not enter production. |

`✅` is deliberately strict. Do not use it for code that merely compiles. A
story is done only when its commit, tests, compatibility check, and required
artifact all exist. Raw XProf traces may live outside Git; commit their manifest,
hash, summarized metrics, and decision.

### Gate labels

| Label | Meaning |
|---|---|
| CPU | Must run in normal CI without TPU hardware. |
| TPU-C | Manual TPU compile or short correctness validation; no performance claim. |
| TPU-M | Manual measured v5e-8 benchmark with warm-up and three trials. |
| EXP | Isolated experiment. It cannot be imported by production modules or become a required dependency. |

### Tracker update rules

1. Change a row to 🟡 only on the working branch.
2. Implement only the scope in that row; include its focused tests in the same
   commit.
3. Run the story's focused test plus the stable regression gate.
4. Set the row to ✅ in that same commit when all assigned gates are complete.
   If a manual TPU gate remains, set it to 🟣.
5. Use the later qualification story to commit the normalized TPU report and
   promotion decision. Do not commit multi-gigabyte traces.
6. Never create an unplanned “cleanup” commit inside a PR. Add a new story row
   first or fold the necessary cleanup into the story that requires it.
7. Preserve these commits when merging. Do not squash the PR, because each
   commit is the audit unit and rollback unit.
8. Update the summary counts and milestone table whenever a row changes.

### Initial progress summary

| State | Count |
|---|---:|
| 🟢 Ready | 1 |
| ⬜ Planned | 105 |
| 🟡 In progress | 0 |
| 🟣 TPU validation pending | 0 |
| ✅ Done | 4 |
| ⛔ Blocked | 0 |
| ↩️ Rolled back | 0 |
| ❌ No-go | 0 |

## 2. Commit contract

Every story commit must satisfy these rules:

- One story ID appears in the commit body, for example `Story: G2-044`.
- The first line uses the exact commit subject in the tracker.
- Production code and the focused tests proving it belong in the same commit.
- The stable default config produces the same resolved backend and model
  behavior as before the commit.
- Every new backend, kernel, architecture, dtype policy, or checkpoint behavior
  has an explicit opt-in setting and a fail-fast validation path.
- A disabled feature must not alter the sharding tree, random stream, optimizer
  state, checkpoint metadata, or loaded batches.
- Avoid commits over roughly 400 changed hand-written lines. If a story exceeds
  that size, split it by adding rows before implementation. Generated lock
  files and compact golden fixtures are excluded from this guideline.
- A commit that changes checkpoint state includes forward and backward
  compatibility tests, or explicitly rejects an unsupported migration before
  reading payload arrays.
- A commit that changes distributed semantics includes a single-device oracle
  and a sharding/collective assertion.
- A commit that introduces a kernel includes a native JAX correctness path and
  records requested versus resolved implementation.
- Each commit is independently revertible with `git revert <sha>`. Reverting
  it must leave all earlier stories usable.

The stable regression gate for every commit is:

```bash
python -m pip check
pytest -q
```

During early development, the author may run a focused test first, but the full
gate is required before a PR merges.

## 3. Milestone tracker

| Status | Milestone | Story range | Outcome | Exit gate |
|---|---|---|---|---|
| 🟡 | M0 — Baseline and modernization | G2-001–G2-015 | Reproducible legacy/Gen-2 environments, canonical benchmark tools, and corrected accounting contracts. | Both CPU dependency lanes pass; legacy behavior is frozen; manual v5e environment capture is recorded. |
| ⬜ | M1 — Mesh, sharding, FSDP, checkpoint foundation | G2-016–G2-038 | Exact ICI/DCN mesh planning, logical axes v2, ZeRO-3-style state sharding, and checkpoint v4. | Tiny unsharded/FSDP updates match and same/cross-layout resume passes. |
| ⬜ | M2 — Shared Gen-2 runtime | G2-039–G2-043 | Opt-in `parallel3d` trainer reaches TP1 parity without changing stable routes. | Three-step train/eval/save/resume parity across native, FSDP, and TP1 mesh runtime. |
| ⬜ | M3 — Dense 1.33B TP | G2-044–G2-069 | TP MLP/GQA/Splash, distributed CE, export, and qualified Dense configs. | Exact parameter count and at least one v5e-8 mesh satisfy correctness, HBM, and measured promotion gates. |
| ⬜ | M4 — MoE reference | G2-070–G2-077 | Native Top-2 MoE semantics with dense and ragged oracles. | Exact total/active counts, output/VJP/update parity, and zero dropped tokens. |
| ⬜ | M5 — EP8 | G2-078–G2-085 | Native ragged all-to-all expert parallel execution. | EP1/2/4/8 parity, checkpoint integrity, routing observability, and acceptable communication cost. |
| ⬜ | M6 — Tokamax MoE candidate | G2-086–G2-089 | Public Tokamax ragged-dot candidate is measured without becoming mandatory. | Promote only on exact-shape full-step evidence; otherwise record ❌ and retain native. |
| ⬜ | M7 — Native GDN2 hybrid | G2-090–G2-097 | Native token and chunked GDN2 with packed resets and 3:1 Splash pattern. | Forward, final-state, VJP, packed-reset, parameter-count, and short resume parity. |
| ⬜ | M8 — Distributed GDN2 and experiments | G2-098–G2-103 | TP/remat GDN2 qualification and isolated KDA/GDN experiments. | Native hybrid qualifies independently; private Tokamax code remains quarantined. |
| ⬜ | M9 — Production hardening | G2-104–G2-110 | Fault-tested runbooks, release gates, and reversible rollout. | Selected architecture passes resume soak, release audit, XProf review, and rollback drill. |

Only one shared-foundation milestone should be active at once. After M3, the
MoE line and GDN2 line may proceed independently, but they must not share a
feature PR.

## 4. PR and branch sequence

Use `feature/**` branch names so the current CI workflow runs automatically.
PRs 01–13 are strictly sequential. PRs 14–16 form the MoE line. PRs 17–18 form
the independent GDN2 line and branch from the merge of PR 13.

| PR | Branch | Stories | Parent | Merge condition |
|---:|---|---|---|---|
| 01 | `feature/gen2-01-dependency-lanes` | G2-001–005 | Current stable main | Legacy and Gen-2 CPU lanes resolve reproducibly. |
| 02 | `feature/gen2-02-observability` | G2-006–010 | PR 01 | Benchmark tools use canonical LLaMA and refuse unsafe output paths. |
| 03 | `feature/gen2-03-batch-contract` | G2-011–015 | PR 02 | Unique-token semantics are explicit and legacy behavior remains selectable. |
| 04 | `feature/gen2-04-mesh-plan` | G2-016–020 | PR 03 | Requested mesh is never silently rewritten. |
| 05 | `feature/gen2-05-axis-rules-v2` | G2-021–026 | PR 04 | Parameter and activation sharding contracts are separate and validated. |
| 06 | `feature/gen2-06-fsdp-correctness` | G2-027–032 | PR 05 | Corrected FSDP matches the oracle and proves state sharding. |
| 07 | `feature/gen2-07-checkpoint-v4` | G2-033–038 | PR 06 | Global-state restore works with target shardings and incomplete saves are ignored. |
| 08 | `feature/gen2-08-parallel3d-tp1` | G2-039–043 | PR 07 | New runtime has TP1 parity and remains opt-in. |
| 09 | `feature/gen2-09-tp-mlp` | G2-044–048 | PR 08 | SwiGLU TP1/2/4 output, VJP, and update parity pass. |
| 10 | `feature/gen2-10-tp-attention` | G2-049–054 | PR 09 | GQA/Splash TP has correct masks and no unintended KV all-gather. |
| 11 | `feature/gen2-11-vocab-ce` | G2-055–059 | PR 10 | Vocab-parallel CE is exact and avoids global logits. |
| 12 | `feature/gen2-12-tokamax-public` | G2-060–064 | PR 11 | Public adapters are versioned, typed, optional, and benchmark-only by default. |
| 13 | `feature/gen2-13-dense-1p33b` | G2-065–069 | PR 12 | Dense model and at least one mesh are qualified on v5e-8. |
| 14 | `feature/gen2-14-moe-reference` | G2-070–077 | PR 13 | Native single-device MoE matches the dense oracle. |
| 15 | `feature/gen2-15-ep8` | G2-078–085 | PR 14 | Native EP8 passes parity, no-drop, resume, and communication gates. |
| 16 | `feature/gen2-16-tokamax-ragged` | G2-086–089 | PR 15 | Public ragged-dot decision is recorded; merging does not imply promotion. |
| 17 | `feature/gen2-17-gdn2-native` | G2-090–097 | PR 13 | Native GDN2 hybrid passes its independent correctness contract. |
| 18 | `feature/gen2-18-gdn2-distributed` | G2-098–103 | PR 17 | Distributed native GDN2 qualifies; private experiments remain isolated. |
| 19 | `feature/gen2-19-release-hardening` | G2-104–110 | PR 13 plus each selected architecture branch | Stable defaults pass; each promoted architecture has its own release verdict. |

PR 16 and the private experiment portion of PR 18 are optional. A ❌ no-go
decision is a valid completion result and must not block the native runtime.

## 5. Detailed commit stories

Within each PR, stories are executed in table order unless a row explicitly
states otherwise.

### PR 01 — Reproducible dependency lanes

Outcome: preserve the current pinned environment and add a separately
reproducible Python 3.12 Gen-2 lane.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ✅ | G2-001 | CPU | `build(gen2): capture immutable runtime manifests` | Add `scripts.capture_environment` and a versioned JSON schema covering Python, JAX, jaxlib, libtpu, Flax, Optax, Orbax, Tokamax, XProf, Grain, devices, Git SHA, dirty state, and config digest. A deterministic CPU fixture passes `tests/gen2/test_environment_manifest.py`. |
| ✅ | G2-002 | CPU | `build(gen2): add the legacy dependency lock lane` | Add legacy input/constraint files under `requirements/inputs` and `requirements/locks` without changing default package versions. A clean Python 3.12 CPU install passes `pip check` and the existing stable tests. |
| ✅ | G2-003 | CPU | `build(gen2): add the modern gen2 dependency input` | Add an opt-in Gen-2 input pinning the approved Python/JAX/Flax/Optax/Orbax/Tokamax/XProf/Grain/Pydantic line from the design plan. Importing LaughLM without the Gen-2 extra must still work. |
| ✅ | G2-004 | CPU | `build(gen2): make hashed lock generation reproducible` | Add a documented lock-generation command or script that produces CPU and TPU-v5e locks with hashes, records the resolver version, and refuses an unpinned direct dependency. Two runs from the same input are byte-identical. |
| 🟢 | G2-005 | CPU | `ci(gen2): test legacy and modern dependency lanes` | Extend CI with separate legacy and Gen-2 CPU jobs. Both run `pip check`, config loading, checkpoint metadata tests, and the stable suite. Gen-2 import/capability tests skip Tokamax cleanly when its extra is absent. |

PR exit gate:

- Current production config resolves identically in the legacy lane.
- Gen-2 dependencies do not replace stable dependencies implicitly.
- Capture one manual v5e-8 environment manifest using the exact command in the
  implementation plan before M0 is marked complete.

Rollback: revert PR 01; no model, config, checkpoint, or runtime code depends on
the new lane yet.

### PR 02 — Canonical observability and benchmark tools

Outcome: all later decisions use one safe benchmark contract and the maintained
`LaughLM/model/llama` path.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-006 | CPU | `fix(profiling): make xprof capability detection explicit` | Repair the programmatic profiler so disabled, unavailable, and active states are distinct. CPU tests prove disabled mode is a no-op and requested-but-unavailable mode fails before training instead of silently pretending to profile. |
| ⬜ | G2-007 | CPU | `feat(bench): define immutable benchmark result records` | Add result schemas for config/environment digests, compile time, warm-up, measured steps, throughput, HBM, collectives, requested/resolved kernels, and trial identity. Output directories must be new and production checkpoint roots are rejected. |
| ⬜ | G2-008 | CPU | `feat(bench): add the canonical llama synthetic harness` | Implement `python -m scripts.benchmark_gen2` using the maintained LLaMA factory, seeded synthetic token IDs, fixed warm-up/measure phases, and no dataset or production checkpoint mutation. Tiny CPU execution produces a valid result record. |
| ⬜ | G2-009 | CPU, TPU-C | `feat(bench): add compile and collective inspection tools` | Implement `scripts.compile_gen2` and `scripts.inspect_collectives` to lower/compile a train step, save HLO metadata, count expected collectives, and detect a reduction inside the accumulation scan. CPU lowering tests pass; a v5e compile artifact is the manual gate. |
| ⬜ | G2-010 | CPU, TPU-C | `feat(bench): add exact-shape kernel and validation CLIs` | Implement `scripts.benchmark_kernels` and `scripts.validate_gen2` with seeded inputs, forward-and-VJP mode, native-reference comparison, and collision-safe artifact paths. Add the v5e command runbook without running heavy training. |

PR exit gate:

- Every tool emits the same artifact envelope and includes environment/config
  digests.
- Benchmark tools cannot point at `checkpoints/production`.
- The legacy `scripts/benchmark_train_step.py` is labeled historical because it
  uses the legacy GPT model.

Rollback: remove the new commands and restore the previous profiler adapter;
training routes are untouched.

### PR 03 — Batch, replica, and token-accounting contract

Outcome: define unique examples independently from tensor replicas and preserve
legacy runs through an explicit contract version.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-011 | CPU | `feat(config): version the distributed batch contract` | Add `runtime.batch_contract` with `legacy_v1` and `unique_replica_v2` plus `micro_batch_per_replica`. Existing configs with no new field resolve to legacy behavior; `parallel3d` requires v2. Validation rejects ambiguous use with `micro_batch_per_device`. |
| ⬜ | G2-012 | CPU | `feat(runtime): centralize batch geometry calculations` | Add a pure geometry helper returning unique replicas, physical replicas, local/global batch, accumulation shape, and tokens/update. Dense/GDN use data times FSDP; MoE may include expert before dispatch; tensor never counts unique samples. |
| ⬜ | G2-013 | CPU | `fix(runtime): make token counters host safe` | Keep optimizer step on device while moving cumulative tokens and per-step increments to checked host-side int64 state. Add overflow, resume, and formatting tests without enabling JAX x64 globally. |
| ⬜ | G2-014 | CPU | `feat(data): add optional distributed sample identity audits` | Add seeded sample IDs or deterministic batch positions to validation-only batches. A collector detects duplicate, missing, and multiply counted examples across simulated data/FSDP/tensor/expert coordinates. Production payloads remain unchanged when disabled. |
| ⬜ | G2-015 | CPU | `test(runtime): freeze legacy and v2 accounting semantics` | Add table-driven tests for PMAP, current FSDP, corrected FSDP, TP, and EP geometries at sequence lengths 2048 and 8192. Label the 41,923 tokens/s historical result as `legacy/pre-correction` in reports. |

PR exit gate:

- Existing configs retain byte-for-byte resolved batch geometry where promised.
- Every Gen-2 config logs physical replicas, unique replicas, and exact
  tokens/update before model initialization.
- No token count uses int32 cumulative storage.

Rollback: select `legacy_v1` or revert PR 03. No sharding behavior changes in
this PR.

### PR 04 — Exact ICI/DCN mesh planning

Outcome: replace dimension rewriting with an immutable, validated mesh plan.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-016 | CPU | `feat(mesh): add the gen2 mesh plan schema` | Add `MeshPlan` and configuration types for separate ICI/DCN data, FSDP, tensor, and expert dimensions. Parsing does not create devices and does not alter the existing mesh function. |
| ⬜ | G2-017 | CPU | `feat(mesh): validate mesh products and inferred dimensions` | Require exact ICI/device-per-slice and DCN/slice products, support at most one documented `-1` per vector, and reject zero, multiple inferred values, and non-divisible products. Requested positive sizes are never rewritten. |
| ⬜ | G2-018 | CPU | `feat(mesh): enforce topology roles for tensor and expert axes` | Make TP and EP ICI-only by default, reject physical sequence/pipeline axes in Gen-2 v1, and require an explicit experimental waiver before DCN TP/EP. Dense and GDN require expert size one. |
| ⬜ | G2-019 | CPU, TPU-C | `feat(mesh): build deterministic topology aware device meshes` | Construct the JAX mesh from `MeshPlan` using stable axis order `data,fsdp,tensor,expert` and deterministic device coordinates. Fake-device tests cover one/multiple slices; a v5e-8 compile confirms the physical order. |
| ⬜ | G2-020 | CPU | `feat(metadata): persist resolved mesh topology manifests` | Record requested/resolved ICI/DCN factors, process/device coordinates, topology waiver, device kinds, and a mesh digest. Metadata round-trips and two different layouts cannot share the same digest. |

PR exit gate:

- Invalid products fail before model or data initialization.
- A requested size is either honored exactly or rejected.
- Existing PMAP/FSDP mesh construction remains available and unchanged.

Rollback: `parallel3d` is not routed yet, so reverting PR 04 removes only unused
Gen-2 mesh code.

### PR 05 — Logical-axis v2 and sharding plans

Outcome: separate parameter and activation semantics and generate auditable
sharding trees.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-021 | CPU | `feat(model): add explicit intermediate size contracts` | Add optional `model.intermediate_size`, preserve the current implicit formula when absent, and thread the value through LLaMA config/factory. Exact tiny, Dense 1.33B, MoE, and GDN candidate parameter calculators are tested. |
| ⬜ | G2-022 | CPU | `feat(sharding): define logical axis rules v2` | Add typed v2 activation and parameter axis names with scalar, tuple/list, or null mappings. Unknown physical/logical axes, unsupported combinations, and duplicate use on one array fail with a path-specific error. |
| ⬜ | G2-023 | CPU | `feat(sharding): translate legacy axis rules deterministically` | Add a one-way translator used only by current PMAP/FSDP configs. Golden tests prove all existing configs resolve to their previous partition specs; ambiguous legacy names are rejected for `parallel3d`. |
| ⬜ | G2-024 | CPU | `refactor(model): split parameter and activation annotations` | Replace overloaded `embed`/`heads` annotations in maintained LLaMA modules with v2 semantic names while retaining a legacy adapter. Initialization and one tiny forward pass are numerically unchanged. |
| ⬜ | G2-025 | CPU | `feat(sharding): generate model state and input sharding plans` | Add `ShardingPlan` generation for params, gradients, optimizer leaves, inputs, recurrent state, and scalar state from abstract trees. Emit an auditable path-to-PartitionSpec report and digest before allocation. |
| ⬜ | G2-026 | CPU | `test(sharding): enforce dimension and replication invariants` | Add table tests for Q/KV head, MLP, vocab, FSDP, expert, and batch divisibility. Reject unexpected replicated array-valued optimizer leaves and illegal reuse of one physical axis twice in an array spec. |

PR exit gate:

- Parameter and activation names cannot be confused.
- Every abstract leaf has a resolved sharding and report entry.
- Legacy config golden files remain unchanged.

Rollback: use axis-rules version 1 and the legacy translator; no trainer uses v2
yet.

### PR 06 — Correct FSDP and optimizer sharding

Outcome: establish measured ZeRO-3-style behavior before adding TP.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-027 | CPU, TPU-C | `fix(fsdp): shard unique batches over data and fsdp` | In the v2 contract, place batch/example dimensions over both data and FSDP and size the loader with their product. Sample-identity tests prove each expected example appears exactly once; legacy_v1 remains selectable. |
| ⬜ | G2-028 | CPU | `fix(fsdp): reduce accumulated gradients exactly once` | Accumulate weighted loss sums, valid-token counts, and FP32 gradient buffers inside `lax.scan`; perform replica reduction and normalization once outside it. Tiny GA1/GA4 results match a concatenated-batch oracle. |
| ⬜ | G2-029 | CPU, TPU-C | `feat(fsdp): constrain persistent gradient shardings` | Apply output sharding constraints matching persistent parameter shards so backward induces reduce-scatter rather than retained full gradients. Lowered-program tests and the collective inspector reject a data reduction inside the scan. |
| ⬜ | G2-030 | CPU | `fix(optimizer): compute global norm and clipping after reduction` | Compute the norm from globally reduced, FSDP-sharded gradients, clip once, and update once. Simulated replica tests match an unsharded optimizer update and detect per-replica clipping. |
| ⬜ | G2-031 | CPU | `feat(optimizer): derive and assert optimizer state shardings` | Build abstract Optax state with `jax.eval_shape`, assign parameter shardings to moments, replicate only scalar state, and fail startup when an array moment unexpectedly uses full replication. Cover FP32 and optional BF16 first moments. |
| ⬜ | G2-032 | CPU, TPU-C | `test(fsdp): qualify zero3 style state and update semantics` | Add end-to-end tiny loss/gradient/update parity, sharding-tree snapshots, sample coverage, and HLO assertions. Manual v5e evidence must show layer-scoped all-gather/reduce-scatter and state HBM scaling before using “ZeRO-3-style” in release docs. |

PR exit gate:

- Tiny update relative error satisfies the design-plan thresholds.
- No persistent full parameter, gradient, or Adam moment copy is observed.
- Existing FSDP users receive an explicit migration note and can retain
  `legacy_v1`.

Rollback: switch the affected config to `legacy_v1` or revert PR 06. PMAP is
untouched.

### PR 07 — Checkpoint v4 and topology-changing restore

Outcome: make the mathematical global state canonical across mesh layouts.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-033 | CPU | `feat(checkpoint): define gen2 checkpoint metadata v4` | Extend metadata with architecture family/version, axis-rules version, mesh/sharding digests, requested/resolved kernels, environment digest, host token count, RNG, and data cursor. Older versions still parse through explicit migration code. |
| ⬜ | G2-034 | CPU | `feat(checkpoint): build abstract target state trees` | Construct shape/dtype/NamedSharding targets for params, Optax state, counters, RNG, and optional recurrent state without allocating full arrays. Unit tests compare every target path with `ShardingPlan`. |
| ⬜ | G2-035 | CPU, TPU-C | `feat(checkpoint): restore global arrays into target shardings` | Restore through public Orbax APIs using the abstract target, first for same layout and then changed data/FSDP/TP layout. Unsupported shape surgery fails before payload restore. |
| ⬜ | G2-036 | CPU | `fix(checkpoint): ignore interrupted asynchronous saves` | Add explicit save-completion state and selection rules so an interrupted async directory is never chosen as latest. Fault fixtures cover missing metadata, partial payload, stale temporary files, and retention. |
| ⬜ | G2-037 | CPU | `feat(checkpoint): persist process independent data and rng state` | Save the logical next-sample position, shard/source cursor, packing state, host token count, and named RNG streams independently of process-local device IDs. Restored next-batch IDs match the uninterrupted run. |
| ⬜ | G2-038 | CPU, TPU-C | `test(checkpoint): add same and cross layout resume equivalence` | Implement `scripts.checkpoint_gen2` and tests for sync/async same-layout resume plus FSDP-to-TP layout change. Step, tokens, RNG, cursor, and the next three losses must match within dtype tolerance. |

PR exit gate:

- Legacy checkpoints remain restorable where their contract permits.
- Incomplete steps are never selected.
- A manual v5e restore from D4/F2/T1 to D2/F2/T2 produces the required
  equivalence report.

Rollback: retain checkpoint v3 writing and v4 reading support while reverting
v4 as the write default. Never delete old checkpoint directories.

### PR 08 — Opt-in `parallel3d` runtime at TP1

Outcome: introduce one shared Gen-2 trainer without duplicating model equations.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-039 | CPU | `feat(runtime): route the parallel3d backend explicitly` | Register `runtime.backend=parallel3d` in schema and entrypoint routing. It requires batch contract v2, axis rules v2, and a valid MeshPlan. Existing `pmap` and `fsdp` classes and defaults are unchanged. |
| ⬜ | G2-040 | CPU | `feat(runtime): add the mesh trainer lifecycle` | Add `MeshTrainer` initialization, abstract model/state construction, sharding-plan application, optimizer creation, logging, and clean shutdown. It uses maintained LLaMA modules and does not subclass or copy MaxText. |
| ⬜ | G2-041 | CPU | `feat(runtime): place global batches with named shardings` | Convert host batches to global arrays over data times FSDP, preserve tensor replication, support prefetch, and expose sample-ID auditing. Input shape and sharding are logged before compilation. |
| ⬜ | G2-042 | CPU | `feat(runtime): add mesh train eval and checkpoint steps` | Add the TP1 jitted train/eval functions using the shared loss contract, accumulation semantics, checkpoint v4, and stable metrics. No MoE/GDN/TP specialization enters this commit. |
| ⬜ | G2-043 | CPU, TPU-C | `test(runtime): prove parallel3d tp1 parity and resume` | Add tiny configs and compare three updates across unsharded native, corrected FSDP, and `parallel3d` TP1. A manual v5e compile/save/resume smoke is required; the backend remains marked experimental afterward. |

PR exit gate:

- TP1 outputs, gradients, updates, counters, and resumed losses meet thresholds.
- Stable trainers instantiate the same classes as before.
- `parallel3d` is unusable without explicit v2 contracts.

Rollback: set backend to `fsdp` or `pmap`. The new trainer has no automatic
selection path.

### PR 09 — Tensor-parallel SwiGLU

Outcome: add classic column/row TP while keeping residual activations replicated
over tensor ranks.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-044 | CPU | `feat(tp): add explicit tensor collective layout helpers` | Add named helpers for column output, row input, replicated residual, and TP reduction constraints. Helpers validate mesh context and become no-ops at TP1; tests inspect their partition specs. |
| ⬜ | G2-045 | CPU | `feat(tp): shard swiglu gate and up projections` | Partition gate/up weights over MLP width, produce local intermediates, and preserve activation/dtype semantics. TP1 and simulated TP2/4 forward/VJP results match dense projections. |
| ⬜ | G2-046 | CPU, TPU-C | `feat(tp): shard the swiglu down projection` | Partition the down projection input over TP and constrain its output to the replicated residual layout, inducing the required reduction. HLO inspection expects one TP reduction and rejects an intermediate all-gather. |
| ⬜ | G2-047 | CPU | `feat(tp): preserve residual norm and remat contracts` | Make norm/residual boundaries explicit, retain FP32 statistics where configured, and ensure block remat does not duplicate collectives. Existing TP1 block numerics and initialization keys are unchanged. |
| ⬜ | G2-048 | CPU, TPU-C | `test(tp): qualify swiglu tp1 tp2 and tp4 updates` | Add exact-shape MLP/block tests, sharding-tree goldens, one-update parity, and v5e compile configs for TP1/2/4. Record collectives and HBM; do not make a speed claim yet. |

PR exit gate:

- TP1 is numerically unchanged.
- TP2/4 meet output, VJP, and update thresholds.
- Residual arrays are replicated over TP in this release.

Rollback: set tensor size one. No parameter conversion is required because
checkpoints store global arrays.

### PR 10 — Tensor-parallel GQA and Splash

Outcome: distribute Q32/KV8 attention without global KV expansion or hard-coded
data-only shard maps.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-049 | CPU | `refactor(attention): expose separate q k and v projections` | Add a separate-projection maintained path while retaining fused QKV for stable TP1 configs. Parameter conversion tests prove round-trip equivalence and fused QKV is rejected for TP greater than one initially. |
| ⬜ | G2-050 | CPU | `feat(tp): shard gqa query and kv heads locally` | Shard Q and KV head dimensions over tensor ranks, validate divisibility for Q32/KV8, and repeat/group KV only inside each local rank when required. Tests cover TP1/2/4/8 mappings. |
| ⬜ | G2-051 | CPU, TPU-C | `feat(tp): shard attention output projection reduction` | Partition O-projection input by query heads and output by FSDP-compatible model width, returning replicated residuals through one TP reduction. HLO checks reject a full-head all-gather before O projection. |
| ⬜ | G2-052 | CPU, TPU-C | `feat(splash): generate mesh aware shard map contracts` | Replace hard-coded data-only specs with generated data/FSDP/tensor specs for global Q/K/V shapes. Requested Splash with an unsupported layout fails under `fallback=error` and records requested/resolved implementations. |
| ⬜ | G2-053 | CPU | `fix(attention): preserve causal and packed masks under tp` | Make causal, padding, and packed segment masks shard-safe at sequence 2048 and 8192. Compare native attention and Splash on tiny packed examples, including document boundaries and GQA. |
| ⬜ | G2-054 | CPU, TPU-C | `test(attention): qualify gqa splash tp layouts` | Add Q32/KV8, head-dim 64 forward/VJP tests and TP1/2/4/8 compile cases. Manual artifacts must show local heads, no unintended KV all-gather, expected O reduction, and no silent fallback. |

PR exit gate:

- Native attention remains the correctness reference.
- Packed documents never attend across segment boundaries.
- Splash TP promotion waits for full-step measurements, not attention-only
  latency.

Rollback: use TP1/fused QKV or choose native attention explicitly. Checkpoint
conversion between fused and separate QKV is tested and reversible.

### PR 11 — Native vocab-parallel linear cross-entropy

Outcome: eliminate the global logits tensor while preserving exact LaughLM loss
semantics.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-055 | CPU | `refactor(loss): unify loss options across all trainers` | Introduce one typed loss-options object carrying backend, chunk size, z-loss, ignore index, remat, and fallback policy. PMAP, FSDP, and MeshTrainer resolve the same values; fix the current FSDP omission. |
| ⬜ | G2-056 | CPU | `feat(loss): add vocabulary ownership and local logits` | Define deterministic vocabulary intervals, padding behavior, and local tied/untied LM-head contractions for TP1/2/4/8. Every target ID has exactly one owning rank. |
| ⬜ | G2-057 | CPU | `feat(loss): compute distributed logsumexp and target logits` | Reduce local maxima and exponential sums across TP, select the owning target logit, and form exact negative log likelihood without materializing global `[B,T,V]` logits. FP32 random tests match dense CE. |
| ⬜ | G2-058 | CPU | `feat(loss): preserve ignore masking z loss and tied weights` | Apply ignore masks before denominator reduction, derive z-loss from the distributed global logsumexp, and share the sharded embedding safely. Tests cover all-ignored, boundary IDs, non-tile-aligned vocab, and z-loss on/off. |
| ⬜ | G2-059 | CPU, TPU-C | `test(loss): qualify vocab parallel ce memory and gradients` | Compare dense, chunked, and distributed output/VJP for vocab 32064 and exact token shapes. Lowered/HBM evidence must show no global logits buffer before the backend is enabled for Dense qualification. |

PR exit gate:

- Loss, z-loss, gradient, and valid-token denominator meet thresholds.
- TP1 native chunked CE remains available.
- Tokamax CE is not used to gather a sharded LM head.

Rollback: select `distributed_vocab=false` at TP1. TP greater than one must fail
rather than gather silently.

### PR 12 — Typed public Tokamax adapters

Outcome: replace speculative probing with optional, version-checked public API
adapters.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-060 | CPU | `feat(kernels): add a public capability registry` | Add a lazy registry that imports only documented top-level Tokamax symbols, records package version/revision, validates supported signatures, and reports unavailable capabilities without breaking native JAX. |
| ⬜ | G2-061 | CPU | `refactor(kernels): quarantine speculative fused op probes` | Remove generic signature guessing from production resolution. Move private or uncertain probes under `scripts/experiments` and add a static test forbidding `tokamax._src` imports anywhere under `LaughLM`. |
| ⬜ | G2-062 | CPU, TPU-C | `feat(kernels): add a typed tokamax linear ce adapter` | Wrap public `linear_softmax_cross_entropy_loss` with explicit implementation/reduction mapping. Restrict it to compatible TP1 semantics, compare output/VJP to native, and reject unsupported bias/z-loss/ignore combinations. |
| ⬜ | G2-063 | CPU, TPU-C | `feat(kernels): add a typed tokamax attention adapter` | Wrap public `dot_product_attention` for Q32/KV8 with explicit sharding, causal mask, dtype, precision, and implementation mapping. No fallback occurs when `fallback=error`. |
| ⬜ | G2-064 | CPU, TPU-M | `test(kernels): benchmark and record public tokamax candidates` | Add exact CE/attention benchmark cases, three-repeat result aggregation, capability/fallback tests, and a decision-report template. Both adapters remain off unless full-step promotion thresholds are later met. |

PR exit gate:

- LaughLM imports and trains natively without Tokamax installed.
- Production package code contains no private Tokamax import.
- Requested and resolved kernels are visible in config, logs, manifests, and
  checkpoints.

Rollback: set all kernel backends to native or uninstall the optional Tokamax
extra.

### PR 13 — Dense 1.33B qualification

Outcome: deliver the first complete Gen-2 model candidate.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-065 | CPU | `feat(config): add dense 1p33b gen2 configurations` | Add complete validation and v5e-8 configs for D0–D5 at sequence 2048 plus approved 8192 rows. Use d_model 2048, 24 layers, Q32/KV8, intermediate 6912, vocab 32064, tied embeddings, and explicit v2 contracts. |
| ⬜ | G2-066 | CPU | `test(model): lock the dense 1p33b parameter contract` | Assert exactly 1,336,641,536 trainable parameters with per-component subtotals. Fail config/model construction if implicit dimensions drift; test tied versus untied head accounting separately. |
| ⬜ | G2-067 | TPU-C, TPU-M | `feat(bench): qualify the dense v5e8 mesh matrix` | Compile D0–D5, validate the tiny mesh oracle, then run three measured trials per eligible row using the plan commands. Commit normalized JSON/Markdown summaries, environment/config hashes, collectives, HBM, and trial variance. |
| ⬜ | G2-068 | CPU, TPU-C | `feat(export): export globally sharded dense checkpoints` | Gather/stream canonical global parameters for Hugging Face export without assuming one host owns full arrays. Validate Q/K/V conversion, tied embeddings, logits, and generated config from a cross-layout checkpoint. |
| ⬜ | G2-069 | TPU-M | `docs(release): record the dense promotion decision` | Fill the acceptance report with correctness, throughput, MFU, peak HBM, compile time, collective analysis, resume, and export results. Select one candidate only if every gate passes; otherwise mark Dense Gen-2 no-go without changing stable defaults. |

PR exit gate:

- Exact model count passes.
- At least one v5e-8 layout has no unexplained collective/full state replica and
  passes checkpoint/export integrity.
- Performance is based on three trials with coefficient of variation at most
  3%.

Rollback: return to `pmap` or current `fsdp` configs. Dense Gen-2 checkpoints
remain readable as global checkpoint v4 state.

### PR 14 — Native Top-2 MoE reference

Outcome: establish LaughLM-owned routing semantics before introducing expert
parallel communication.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-070 | CPU | `feat(moe): add the opt in moe configuration contract` | Add fields for 8 experts, Top-2, expert intermediate 2048, router dtype, no-drop ragged capacity, auxiliary coefficients, expert backend, dispatch backend, and shared-weight strategy. Dense configs are unaffected. |
| ⬜ | G2-071 | CPU | `feat(moe): implement deterministic fp32 top2 routing` | Compute FP32 router logits, use stable Top-2 selection, normalize selected weights, define deterministic tie behavior, and mask ignored/padding tokens. Golden cases cover ties and extreme logits. |
| ⬜ | G2-072 | CPU | `feat(moe): add router auxiliary and z losses` | Implement load-balancing loss and router z-loss as separately logged terms with explicit scaling. Finite-difference/VJP tests match a tiny NumPy-style oracle and disabled coefficients contribute exactly zero. |
| ⬜ | G2-073 | CPU | `feat(moe): expose complete routing health metrics` | Record tokens/assignments per expert, max/mean ratio, entropy, dead/empty experts, overflow/drop count, and valid-token denominator. Metric aggregation is correct across accumulation steps and data replicas. |
| ⬜ | G2-074 | CPU | `test(moe): add a dense all expert routing oracle` | Build a tiny one-hot all-expert einsum reference with forced balanced, skewed, empty-expert, and tied-logit routes. It is test-only and defines forward, combine, loss, and VJP truth. |
| ⬜ | G2-075 | CPU | `feat(moe): add stable ragged dispatch metadata` | Flatten Top-2 assignments, stable-sort by expert/token/slot, compute group sizes and inverse permutations, and retain combine weights. Round-trip tests recover exact assignment order with empty experts. |
| ⬜ | G2-076 | CPU, TPU-C | `feat(moe): execute native ragged expert mlps` | Run local expert SwiGLU with public native JAX ragged operations, no capacity dropping, and FP32 accumulation where needed. Output/VJP matches the dense oracle on tiny shapes. |
| ⬜ | G2-077 | CPU, TPU-C | `feat(model): integrate moe decoder blocks and exact counts` | Integrate MoE under `ffn_type=moe` in maintained LLaMA, combine routed outputs, and assert 2,733,737,984 total plus 921,798,656 active parameters. Tiny train/update/checkpoint tests pass at EP1. |

PR exit gate:

- Forced-route output, losses, VJP, and update match the dense oracle.
- Baseline drops zero assignments.
- MoE remains a separate opt-in architecture and no EP collective exists yet.

Rollback: set `ffn_type=swiglu`. Dense checkpoint identity is unaffected.

### PR 15 — Native expert parallelism through EP8

Outcome: route tokens to one expert per v5e-8 rank using public JAX primitives.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-078 | CPU | `feat(ep): add role aware expert axis layouts` | Define shared-layer and expert-region layout transitions. Shared layers may treat expert ranks as unique batch/FSDP participants; expert regions map the physical axis to expert parameters and routed token ownership. |
| ⬜ | G2-079 | CPU, TPU-C | `feat(ep): shard expert parameters over the expert axis` | Place one expert per rank at EP8 and two per rank at EP4 while preserving global arrays/checkpoints. Sharding tests cover expert counts divisible by EP and reject unsupported layouts. |
| ⬜ | G2-080 | CPU, TPU-C | `feat(ep): dispatch ragged assignments with all to all` | Use public `jax.lax.ragged_all_to_all` to send sorted assignments to owning ranks, including group metadata and source positions. Balanced, skewed, and empty destination fixtures round-trip without drops. |
| ⬜ | G2-081 | CPU, TPU-C | `feat(ep): execute local experts after dispatch` | Run only rank-local expert parameters over received ragged rows, preserve BF16/FP32 policy, and expose sent/received counts. EP1 and simulated EP2 results match the single-device ragged path. |
| ⬜ | G2-082 | CPU, TPU-C | `feat(ep): return and combine expert outputs` | Add reverse ragged all-to-all, inverse permutation, Top-2 weighted combine, and residual layout restoration. Forward/VJP covers zero-row experts and duplicated token assignments. |
| ⬜ | G2-083 | CPU, TPU-M | `feat(ep): support replicated and sharded shared weights` | Implement explicit `expert_as_fsdp_for_shared` strategies with separate sharding reports. Benchmark both on v5e-8; neither is selected by assumption. |
| ⬜ | G2-084 | CPU, TPU-C | `feat(ep): checkpoint and report expert parallel state` | Extend checkpoint/manifests with EP layout and routing/communication metrics while keeping mathematical state topology-independent. EP8-to-EP4 restore and next-loss parity are tested where memory permits. |
| ⬜ | G2-085 | TPU-C, TPU-M | `test(ep): qualify ep1 ep2 ep4 and ep8` | Add M0–M3 configs and run forced-route correctness followed by three exact full-step trials. Commit zero-drop evidence, all-to-all bytes/time, router health, HBM, throughput, and a native EP8 decision. |

PR exit gate:

- EP1/2/4/8 meet output/VJP/update thresholds.
- No assignment is dropped, including skewed/empty-expert cases.
- Routing and all-to-all cost are visible in every measured result.

Rollback: select EP1 and the single-device ragged backend. Checkpoint global
expert arrays remain valid.

### PR 16 — Public Tokamax ragged-dot candidate

Outcome: measure public Tokamax expert GEMMs without depending on private GMM or
fused MoE APIs.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-086 | CPU, TPU-C | `feat(kernels): add a public tokamax ragged dot adapter` | Wrap only the top-level public `ragged_dot` API with explicit implementation, dtype, precision, and group-size validation. Forward/VJP matches native ragged dot for balanced, skewed, and empty groups. |
| ⬜ | G2-087 | CPU | `feat(autotune): key and persist exact shape tuning records` | Define tuning keys from operation, full/local shapes, dtype, mesh, JAX/libtpu/Tokamax revisions, and implementation. Reject stale/mismatched records and serialize deterministic winners. |
| ⬜ | G2-088 | TPU-M | `feat(bench): compare tokamax ragged dot in full moe steps` | Benchmark public XLA/Mosaic candidates for the exact 16,384-token, 32,768-assignment EP8 case and M0–M3 full steps. Use three trials and native verification; microbenchmark-only wins are insufficient. |
| ⬜ | G2-089 | TPU-M | `docs(kernels): record the ragged dot promotion verdict` | Commit the normalized report and choose `USE` or `NO-GO`. Promotion requires at least 5% full-step throughput or 10% HBM improvement with at most 2% throughput regression. Native remains default on ties/failures. |

PR exit gate:

- No `tokamax._src` import enters production.
- Missing Tokamax or missing tuning records select native or fail according to
  explicit config policy.
- ❌ is an acceptable final status for G2-089.

Rollback: set grouped matmul backend to `native_ragged_dot` or remove the
optional kernel extra.

### PR 17 — Native GatedDeltaNet2 and 3:1 Splash hybrid

Outcome: create an architecture branch independent from MoE.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-090 | CPU | `feat(gdn2): add the hybrid architecture contract` | Add opt-in GDN2 fields and an explicit layer pattern. Validate 24 layers with zero-based Splash layers 3, 7, 11, 15, 19, and 23; reject simultaneous MoE/GDN2 in this branch. |
| ⬜ | G2-091 | CPU | `feat(gdn2): add projection and recurrent initialization` | Implement Q/K/V/Z and beta/decay projections, A-log/dt-bias state, documented dimensions K16/V32/D128, and deterministic initialization. Parameter-tree snapshots are stable. |
| ⬜ | G2-092 | CPU | `feat(gdn2): add a causal depthwise conv1d oracle` | Implement the kernel-size-4 causal depthwise convolution with explicit initial/final cache and an FP32 test oracle. Token-by-token and full-sequence output/cache agree. |
| ⬜ | G2-093 | CPU | `feat(gdn2): implement the fp32 token recurrence oracle` | Add the literal per-token gated delta update, including beta, decay, state update, output, and final state. This slow path is the semantic reference and test backend only. |
| ⬜ | G2-094 | CPU, TPU-C | `feat(gdn2): add the native chunked training recurrence` | Implement chunk-size-64 recurrence using public JAX operations with custom chunk state only where necessary. Sequence 32/64/128 forward, final-state, and VJP match the token oracle. |
| ⬜ | G2-095 | CPU | `fix(gdn2): reset convolution and delta state for packed documents` | Consume segment/document IDs so neither convolution cache nor recurrent state crosses a packing boundary. Adversarial packed tests compare separately evaluated documents with packed evaluation. |
| ⬜ | G2-096 | CPU | `feat(gdn2): add gated normalization and output projection` | Apply recurrent output normalization, SiLU Z gate, and output projection with the configured precision policy. Test each sublayer and the composed GDN2 block against explicit equations. |
| ⬜ | G2-097 | CPU, TPU-C | `feat(model): integrate the three to one gdn2 splash hybrid` | Insert 18 GDN2 and 6 Splash blocks into maintained LLaMA, add a parameter-count assertion for the selected approximately 1.34B config, and pass tiny forward/VJP/update/save-resume parity. |

PR exit gate:

- Token and chunked recurrence agree in output, final state, and VJP.
- Packed resets are exact.
- MoE code is neither imported nor configured by this architecture branch.

Rollback: select the Dense architecture. GDN2 checkpoints carry a distinct
architecture identifier and cannot be misread as Dense.

### PR 18 — Distributed GDN2 and isolated kernel experiments

Outcome: qualify native TP/remat first, then optionally compare experimental
private kernels in a quarantined process.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-098 | CPU, TPU-C | `feat(gdn2): shard recurrent heads over tensor parallelism` | Partition K/V heads and projection outputs over TP, reduce the output projection into replicated residuals, and validate divisibility. TP1/2/4 output/VJP matches the unsharded native chunked path. |
| ⬜ | G2-099 | CPU, TPU-C | `feat(gdn2): shard recurrent state with explicit precision` | Define recurrent and convolution-cache shardings, retain FP32 accumulation where required, and prove no rank owns an unintended full state. Checkpoint abstract targets include recurrent state when present. |
| ⬜ | G2-100 | CPU, TPU-C | `feat(gdn2): add recurrence aware rematerialization policies` | Add named save points for block input, convolution output, and chunk state. Benchmark no-remat versus block/chunk policies independently and reject accidental nested full remat. |
| ⬜ | G2-101 | CPU, TPU-C | `test(gdn2): qualify distributed updates and resume` | Compare TP1/2/4 hybrid loss, gradients, one update, final state, and save/resume. Inspect collectives and require the same checkpoint integrity as Dense. |
| ⬜ | G2-102 | EXP | `chore(experiments): isolate private tokamax gdn and kda probes` | Add only `scripts.experiments.benchmark_tokamax_private` with an exact revision guard and subprocess boundary. Static CI forbids private imports elsewhere; semantic mismatch stops before timing. |
| ⬜ | G2-103 | TPU-M, EXP | `docs(gdn2): record native and experimental v5e8 decisions` | Run the native 3:1 full-step matrix and, optionally, isolated KDA/GDN probes. Commit separate verdicts: native qualification is independent; private kernels remain experimental even if faster. |

PR exit gate:

- Native distributed GDN2 meets correctness and resume gates.
- Any private Tokamax result is clearly labeled non-production and revision
  pinned.
- Failure or removal of Tokamax cannot affect native GDN2 imports/checkpoints.

Rollback: set tensor size one or return to Dense. Delete no checkpoint; use the
architecture identifier to select compatible restores.

### PR 19 — Production hardening and reversible rollout

Outcome: make promotion evidence-based, fault-tested, and operationally
reversible while keeping stable behavior available.

| Status | ID | Gate | Exact commit subject | Deliverable and acceptance |
|---|---|---|---|---|
| ⬜ | G2-104 | CPU | `feat(release): emit complete gen2 run manifests` | Consolidate environment, config, data, model, mesh, sharding, kernel, checkpoint, and benchmark digests in one versioned manifest. Release audit rejects missing or internally inconsistent fields. |
| ⬜ | G2-105 | CPU | `feat(release): standardize qualification artifact indexes` | Add a small artifact index linking normalized metrics, HLO summary, XProf hash/location, checkpoint audit, trial IDs, and decision report. Raw traces stay out of Git and overwrite is forbidden. |
| ⬜ | G2-106 | CPU, TPU-C | `test(release): inject checkpoint and process failures` | Add safe fault-injection cases for interruption before/during/after async save, stale latest pointer, missing host, and restart with changed mesh. Complete checkpoints recover; incomplete ones are ignored. |
| ⬜ | G2-107 | TPU-C | `feat(release): add bounded resume soak commands` | Add manual commands for multiple save/stop/restore cycles with deterministic next-batch checks and three post-restore losses. Run separately for Dense, MoE, and GDN2 candidates selected for release. |
| ⬜ | G2-108 | CPU, TPU-M | `feat(release): enforce architecture specific promotion gates` | Encode release checks for correctness thresholds, trial variance, no unexplained state replication, checkpoint integrity, MoE no-drop/router health, and GDN final-state/reset parity. One architecture can fail without blocking another. |
| ⬜ | G2-109 | CPU, TPU-C | `feat(release): add explicit runtime and kernel rollback switches` | Validate one-command fallback to stable PMAP/FSDP, native attention/CE/ragged/GDN implementations, TP1, and EP1 where applicable. A rollback drill restores a pre-promotion checkpoint without mutation. |
| ⬜ | G2-110 | CPU | `docs(release): publish gen2 runbooks and final decisions` | Publish operator commands, supported configs, known limits, rollback steps, checkpoint compatibility, dependency locks, and separate Dense/MoE/GDN/kernel verdicts. Update this tracker and the design plan links. |

PR exit gate:

- Current stable PMAP and FSDP tests pass in the legacy dependency lane.
- Every promoted Gen-2 config is explicit and records its resolved implementation.
- Dense, MoE, and GDN2 each have an independent go/no-go record.
- A rollback drill has been performed and documented.

Rollback: choose the last known stable config and dependency lock, disable
`parallel3d`, and restore a compatible checkpoint. Production launch tooling
must never rewrite or delete the previous checkpoint directory.

## 6. Manual TPU validation queue

These are validation stories, not commands for local Windows development. The
full exact command lines and shapes remain in section 10 of the
[implementation plan](GEN2_IMPLEMENTATION_PLAN.md#10-v5e-8-benchmark-and-validation-plan).

| Status | Story | Hardware work to run manually | Evidence to retain |
|---|---|---|---|
| ⬜ | G2-001/005 | Install the hashed v5e lock, run `pip check`, capture eight TPU devices. | Environment JSON and lock/config/Git digests. |
| ⬜ | G2-009/010 | Compile tiny LLaMA and capture a bounded XProf trace. | HLO summary, profiler state, trace hash, no production path touched. |
| ⬜ | G2-027–032 | Run corrected FSDP tiny update and inspect all-gather/reduce-scatter/state HBM. | Parity JSON, collective report, peak HBM, sample coverage. |
| ⬜ | G2-035/038 | Save D4/F2/T1 and restore D2/F2/T2. | State/cursor/RNG comparison and next-three-loss report. |
| ⬜ | G2-043 | Run three-step `parallel3d` TP1 smoke. | Native/FSDP/mesh parity and checkpoint audit. |
| ⬜ | G2-048 | Compile TP1/2/4 SwiGLU. | Partition tree, collective counts, HBM. |
| ⬜ | G2-054 | Compile Q32/KV8 Splash TP1/2/4/8 at 2048 and 8192. | Mask parity, no-KV-gather evidence, resolved backend. |
| ⬜ | G2-059 | Run exact distributed CE shapes. | Loss/VJP parity and no-global-logits HBM/HLO proof. |
| ⬜ | G2-064 | Run public Tokamax CE and attention matrix. | Three-trial kernel and full-step summaries; promotion deferred to evidence. |
| ⬜ | G2-067/069 | Run D0–D5 Dense matrix. | Three trials/config, CV, throughput, MFU, HBM, compile, resume, export verdict. |
| ⬜ | G2-076/077 | Compile native MoE at EP1. | Dense-oracle parity and exact parameter counts. |
| ⬜ | G2-080–085 | Run forced routing and M0–M3 EP matrix. | No-drop proof, router health, all-to-all cost, checkpoint report. |
| ⬜ | G2-086–089 | Run public Tokamax ragged-dot exact shapes. | Native parity, tuning key, full-step decision. |
| ⬜ | G2-094/097 | Validate token versus chunked GDN2. | Output/final-state/VJP/reset report. |
| ⬜ | G2-098–103 | Run distributed native hybrid and optional private subprocess experiment. | Native release report and separately labeled experimental verdict. |
| ⬜ | G2-106–109 | Run fault, soak, release, and rollback drills. | Release audit and architecture-specific signed-off decisions. |

## 7. Promotion gates

Use the exact thresholds from the implementation plan:

| Area | Required go condition | No-go or fallback |
|---|---|---|
| FP32 oracle | Output/loss `rtol <= 1e-5` and `atol <= 1e-6`. | Fix semantics; do not benchmark performance. |
| BF16 TPU | Loss `rtol <= 3e-3` and `atol <= 3e-3`; gradient cosine at least 0.999; update relative L2 at most 5e-3. | Keep native/unsharded reference and block promotion. |
| Samples/tokens | Every expected sample ID exactly once across unique replicas; token count exact. | Return to legacy stable runtime; correct batch contract first. |
| FSDP | Persistent state scales with FSDP and HLO shows scoped gathers/reduce-scatters. | Remain corrected data parallel or smaller model; do not claim ZeRO-3. |
| TP | Correct TP1/2/4 behavior, expected collectives, no unintended KV/global-logit gather. | Use smaller TP factor or TP1. |
| Checkpoint | Step/tokens/RNG/data cursor exact; next three losses within tolerance. | Keep backend experimental and restore prior checkpoint implementation. |
| Performance stability | Three trials with throughput CV at most 3%. | Repeat after eliminating noise; make no promotion decision. |
| Kernel promotion | At least 5% full-step throughput gain, or 10% HBM reduction with no more than 2% throughput loss. | Native remains default; record ❌. |
| MoE | Zero dropped assignments, finite router losses, metrics complete, no dead expert during smoke. | Use Dense or EP1 native reference. |
| GDN2 | Token/chunk output, final state, VJP, and packed resets pass. | Use Dense/Splash; experimental KDA cannot override semantics. |

## 8. Review checklist for every PR

Before requesting review:

- [ ] Every commit maps to exactly one story ID and uses its exact subject.
- [ ] Story rows and progress totals are current.
- [ ] `pytest -q` and `python -m pip check` pass in every affected CPU lane.
- [ ] Current production YAML resolves to the same backend and stable behavior.
- [ ] New code is unreachable unless an explicit Gen-2 setting enables it.
- [ ] Native JAX is present as the correctness/reference implementation.
- [ ] No production module imports `tokamax._src`.
- [ ] Requested and resolved implementations are logged and checkpointed.
- [ ] Checkpoint changes include restore and incomplete-save tests.
- [ ] Distributed changes include sample, sharding, and collective evidence.
- [ ] Manual TPU commands are copied into the PR description, not executed
      locally.
- [ ] Rollback switch and checkpoint implications are documented.
- [ ] No unrelated user changes are reformatted or included.

## 9. Final definition of done

LaughLM Gen-2 is complete only when:

1. G2-001 through G2-069 are ✅ and Dense 1.33B has a documented v5e-8 verdict.
2. The shared runtime can train with DP times FSDP times TP, resume across a
   supported layout change, and export a valid Hugging Face checkpoint.
3. The current stable trainers/configs remain supported and default.
4. Native JAX works without Tokamax installed.
5. MoE is complete only when G2-070 through G2-085 are ✅; G2-086–089 may end
   ✅ with either a promotion or a recorded ❌ candidate verdict.
6. GDN2 is complete only when G2-090 through G2-101 are ✅; G2-102–103 are
   optional experiments and cannot redefine the architecture.
7. Production promotion additionally requires G2-104 through G2-110 for each
   selected architecture.
8. No unresolved checkpoint-integrity, sample-duplication, silent-kernel-
   fallback, or persistent-state-replication issue remains.

Completion is architecture-specific. Dense may be production-ready while MoE
or GDN2 remains experimental. No roadmap item requires shipping all three at
the same time.
