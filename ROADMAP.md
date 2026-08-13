# LaughLM Training Stabilization Roadmap

**Status:** Draft for execution  
**Scope:** Resolve the remaining scan, Grain, checkpointing, and single-VM TPU startup risks in the optimized Llama training path.  
**Starting revision:** `b96d922f78e6591c81c49ecd3417ee90981eff51` on `feature/performance-profiler`.

## Purpose and operating model

This roadmap is intentionally narrower than [`docs/v2_optimization_roadmap.md`](docs/v2_optimization_roadmap.md). It is the execution plan for restoring a **reliable optimized training run** before additional architecture or performance work continues. The current `v5e_pmap_optimized.yaml` configuration enables both scanned Llama layers and the Grain backend, so the scan and batching defects must be treated as release blockers.

> **One-milestone rule:** only one milestone may be active at a time. The next milestone begins only after the current milestone has passed every acceptance gate, its tests are committed, and its completion evidence is recorded in the pull request or issue. Defects discovered in a later milestone are triaged back to their originating milestone rather than patched opportunistically.

| Order | Milestone | Primary outcome | Entry condition | Exit condition |
|---|---|---|---|---|
| M0 | Reproducible baseline | A pinned, repeatable local and TPU validation environment | This roadmap is accepted | Baseline failure and environment are documented |
| M1 | Scanned Llama contract | A valid, tested `nn.scan` carry contract | M0 complete | Scanned and unscanned forward paths agree on a deterministic smoke case |
| M2 | Grain data pipeline | A correctly batched, resumable per-host input pipeline | M1 complete | Grain yields fixed-size batches and restores iterator state |
| M3 | Checkpoint compatibility | Orbax lifecycle proven against the supported dependency set | M2 complete | Composite save, retention, and restore succeed |
| M4 | Single-VM TPU startup | TPU runtime starts without explicit distributed initialization | M3 complete | A short TPU smoke run initializes and executes safely |
| M5 | End-to-end release gate | Optimized training is safe to use and regressions are guarded | M4 complete | Fresh run, resume run, and automated regression suite pass |

## Current implementation status

> An initial remediation patch has been prepared for the scan axes, Grain batching API, Orbax manager construction, and single-VM TPU startup policy. No milestone is complete until the user runs the relevant manual validation gates.


## Known-risk register

| ID | Risk | Current evidence | Owner milestone | Release impact |
|---|---|---|---|---|
| R-01 | The Llama scan wrapper does not maintain the intended stable carry structure. | `LaughLM/model/llama/model.py` calls the scanned layer with five separate arguments while the wrapper returns a two-item result. | M1 | Blocking |
| R-02 | The Grain loader uses the removed `DataLoader(batch_size=...)` API and no batch operation. | `LaughLM/utils/data_factory.py` still constructs `grain.DataLoader(..., batch_size=...)`. | M2 | Blocking when `data_backend: grain` |
| R-03 | Orbax behavior must be verified against the actual pinned package versions. | The manager now uses `CheckpointManagerOptions` and `options=`, but the project does not pin a tested JAX/Flax/Grain/Orbax stack. | M3 | High |
| R-04 | Single-VM TPU startup must continue to rely on JAX device discovery. | `scripts/train_tpu_optimized.py` no longer explicitly initializes distributed JAX. | M4 | High |
| R-05 | There are no focused regression tests for scanned Llama layers or the Grain pipeline. | The current test suite does not cover these two enabled optimized paths. | M5 | High |

## M0 â€” Reproducible baseline and failure capture

**Objective.** Establish the dependency and execution baseline before changing behavior. The connected development environment currently lacks JAX, Flax, Grain, and Orbax, so runtime outcomes cannot be trusted until a named environment is created from a reproducible specification.

| Work item | Deliverable | Acceptance evidence |
|---|---|---|
| Define the supported Python, JAX, `jaxlib`, Flax, Grain, Orbax, Optax, and TPU runtime versions. | A lockfile or a documented constraints file referenced by the installation instructions. | A clean environment installs without solver drift. |
| Record the minimal reproducer for each current failure. | A short command and captured traceback for scan and Grain failures; record the current single-VM TPU startup behavior. | Each failure is reproducible before its milestone begins. |
| Create a lightweight test configuration. | A CPU-safe tiny Llama configuration with deterministic seed, tiny vocabulary, short context, and one or two layers. | Model initialization and a non-scanned forward pass complete. |
| Define evidence storage. | A standard `tests/` location for regression tests and a concise PR checklist template. | Future milestones can link test names and logs rather than prose only. |

**Completion gate.** Commit the version constraints, tiny configuration, and baseline tests or reproducer scripts. Do not modify scan logic, the Grain API call, or TPU startup policy during M0.

## M1 â€” Scanned Llama carry-contract repair

**Objective.** Make `nn.scan` use a single, JAX-compatible carry pytree that is unpacked and returned in exactly the same structure and order on every layer iteration. Flax scan bodies must follow the `(carry, *xs) -> (carry, ys)` contract, and values that are global to all layers must be broadcast rather than scanned along a non-existent layer axis.[1]

**Scope.** This milestone owns `LaughLM/model/llama/model.py`, `LaughLM/model/llama/decoder.py`, and any directly involved scan adapter. It does not change attention math, optimizer behavior, data loading, or checkpoint formats.

| Work item | Implementation requirement | Validation gate |
|---|---|---|
| Freeze the active Llama layer contract. | Reconcile the incident prompt's `freqs_cis` and `deterministic` fields with the current `positions`, `attention_mask`, `kv_cache`, and `mode` interface before coding. Record the selected contract in a code comment and test name. | There is one authoritative layer signature; no parallel legacy signature remains. |
| Create the scan adapter. | Use an explicit named carry type or documented tuple. It must contain the intended five elements in a stable order, and the adapter must return the same five-element structure. Python strings and other non-JAX state must be supplied as static/broadcast inputs or encoded safely; they must not become an unstable scan carry. | `jax.tree_util.tree_structure` of input and output carries matches. |
| Correct scan axes. | Only per-layer parameters are scanned on the layer axis. Shared tensors and control inputs use broadcast semantics, while hidden state and cache state are carried. | `nn.scan` initialization succeeds for the tiny configuration. |
| Preserve train and decode behavior. | Verify cache-free training, cached decode, and the intended deterministic/dropout behavior separately. | Shape, dtype, and cache-tree assertions pass for each mode. |
| Add a regression suite. | Cover scan initialization, one forward pass, and deterministic numerical agreement with the unscanned stack using identical parameters. | The scanned and unscanned logits satisfy a defined tolerance. |

**Completion gate.** The tiny test configuration passes scan initialization and forward execution. For a deterministic configuration, scanned and unscanned outputs are numerically equivalent within the agreed tolerance. The new regression tests run on CPU and are required in continuous integration.

## M2 â€” Grain batching and resumability

**Objective.** Replace the obsolete loader-level `batch_size` parameter with an explicit Grain operations pipeline, while preserving per-host batch sizing and deterministic resume behavior. Grain applies transformations through `operations`, and its `Batch` transform accepts `batch_size` and `drop_remainder` directly.[2]

**Scope.** This milestone owns `LaughLM/utils/data_factory.py` and focused tests/fixtures. It may adjust caller assumptions only where a batch-shape contract requires it.

| Work item | Implementation requirement | Validation gate |
|---|---|---|
| Define batch ownership. | Document that `global_batch_size` is divided by `process_count`, and establish `per_process_batch_size` as the only batch size given to Grain. | One host produces exactly the expected local batch shape. |
| Build the operations pipeline. | Remove `batch_size=` from `grain.DataLoader`. Pass `operations=[grain.Batch(batch_size=self.per_process_batch_size, drop_remainder=True)]` after token-record creation. | Grain initializes without an unexpected-keyword error. |
| Prevent double batching. | Verify source records remain individual `seq_len + 1` token windows until the Batch operation runs. | Returned tensors have exactly `[per_process_batch_size, seq_len + 1]`. |
| Preserve multi-host determinism. | Retain `IndexSampler` sharding and seed semantics; test distinct host shards and intentional final-batch dropping. | Two simulated shards are non-overlapping and fixed-size. |
| Prove checkpoint/resume semantics. | Advance the iterator, serialize its state through the existing wrapper, restore it, and compare the next batch. | The restored iterator yields the same next batch as the uninterrupted iterator. |

**Completion gate.** A focused Grain test proves fixed local batch shape, `drop_remainder=True`, deterministic sharding, and exact next-batch recovery after restore. The native memmap backend must retain its existing test behavior.

## M3 â€” Orbax checkpoint compatibility and lifecycle proof

**Objective.** Verify the already-applied `CheckpointManagerOptions` migration against the supported dependency set and prove that model, optimizer, metadata, and Grain state form a coherent checkpoint lifecycle. Orbax exposes manager behavior through `CheckpointManagerOptions`, and checkpoint saves may be asynchronous, so tests must wait before inspection or restoration.[3]

**Scope.** This milestone owns `LaughLM/utils/checkpoint_factory.py`, dependency constraints established in M0, and checkpoint-focused tests. Avoid switching checkpoint formats unless existing behavior fails validation.

| Work item | Implementation requirement | Validation gate |
|---|---|---|
| Lock the Orbax interface. | Confirm that the pinned Orbax version supports every selected option, including async settings and background deletion. | Manager construction emits no legacy API warning or unsupported-option error. |
| Validate single-item state. | Save and restore a small model/optimizer pytree using the production manager construction. | Restored leaves equal the saved state. |
| Validate composite state. | Include model state, metadata, and Grain iterator state in the production save/restore path. | All items restore with the intended names and types. |
| Validate retention and synchronization. | Save more than `max_to_keep` checkpoints, call `wait_until_finished`, and inspect retained steps. | Exactly the configured retention set remains. |
| Define failure behavior. | Ensure missing/corrupt checkpoint handling falls back or raises a clear error without silently changing training state. | Negative-path tests assert the selected behavior. |

**Completion gate.** A clean process can save a checkpoint, wait for completion, restart, restore the composite state, and continue from the same model and input position. Retention behavior is tested under the pinned Orbax version.

## M4 â€” Single-VM TPU startup hardening

**Objective.** Preserve the removal of the failing explicit multi-host initialization path and make single-VM TPU assumptions visible, tested, and safe.

**Scope.** This milestone owns `scripts/train_tpu_optimized.py`, its launch documentation, and TPU-specific smoke configuration. It does not add multi-host training support.

| Work item | Implementation requirement | Validation gate |
|---|---|---|
| Make the deployment policy explicit. | State in the script documentation that this entrypoint targets one TPU VM and relies on JAX runtime discovery. | The entrypoint contains no `jax.distributed.initialize()` call or manual `tpu_process_addresses` setting. |
| Preserve runtime discovery. | Use `jax.local_device_count()` for device-local batch calculation and JAX-reported process index/count for data sharding. | Startup logs report expected local devices and one resolved process on the target VM. |
| Add preflight checks. | Emit clear diagnostics for zero devices, invalid microbatch divisibility, missing token shards, and unsafe fresh-checkpoint deletion. | Expected configuration errors fail before model initialization. |
| Execute an accelerator smoke run. | Run a bounded training job using the M1/M2/M3 validated configuration and an isolated checkpoint directory. | Initialization, at least one optimizer update, logging, and clean shutdown succeed. |
| Guard against scope creep. | If multi-VM support is required, create a separate RFC and milestone series with an explicit coordination design. | No multi-host flags are silently introduced into this entrypoint. |

**Completion gate.** A fresh single-VM TPU run starts without the former SliceBuilder `INVALID_ARGUMENT` failure, performs a bounded update, and exits with a durable checkpoint when checkpointing is enabled.

## M5 â€” Integrated optimized-run release gate

**Objective.** Demonstrate that the optimized configuration works as a system and make its critical behavior resistant to regression.

**Scope.** This milestone integrates, but does not redesign, the outputs of M0 through M4. It owns `configs/v5e_pmap_optimized.yaml`, release notes, automated checks, and final evidence.

| Work item | Implementation requirement | Validation gate |
|---|---|---|
| Validate a fresh run. | Start from an empty, isolated checkpoint directory with `scan_layers: true`, `data_backend: grain`, and async checkpointing enabled. | The run completes the defined smoke-step count with finite loss. |
| Validate a resume run. | Stop after a checkpoint, create a new process, restore, and continue. | Step count, model state, optimizer state, and next input batch resume coherently. |
| Compare execution modes. | Run the same tiny workload in scanned and unscanned modes under deterministic settings. | Outputs/losses meet the M1 tolerance, and both paths remain usable. |
| Add automation. | Require unit tests for scan, Grain, and checkpoint behavior; make TPU smoke evidence part of release review. | CI blocks regressions before merge. |
| Publish operational notes. | Document supported environment versions, single-VM constraint, launch command, resume command, and known non-goals. | README or dedicated runbook is linked from the project entrypoint. |

**Completion gate.** The optimized configuration completes a fresh bounded run and a restart/resume run on the target single-VM TPU environment. All focused regression tests pass, required operational documentation is merged, and no unresolved blocking risks remain in the register.

## Execution protocol

Every milestone should use a dedicated branch and a small, reviewable pull request. The pull request description must include the milestone identifier, the test command(s), the observed result, and a short rollback note. Avoid bundling unrelated performance optimizations, model changes, or dependency upgrades with a stabilization milestone.

| Required artifact | Standard |
|---|---|
| Issue or PR title | `[M#] concise milestone outcome` |
| Tests | A new regression test for every repaired failure mode, plus relevant existing tests |
| Review evidence | Commands, environment versions, and pass/fail output summarized in the PR |
| Rollback plan | Revert the milestone commit; retain diagnostic logs and the prior working configuration |
| Advancement decision | Explicit sign-off that every completion gate passed before opening the next milestone |

## Start here

**Activate M0 first.** Create the supported environment specification and capture the current scan and Grain failures with the tiny configuration. Once M0's evidence is committed, begin M1; do not begin M2, M3, M4, or M5 in parallel.

## References

[1] [Flax Linen `scan` API](https://flax.readthedocs.io/en/v0.6.11/api_reference/_autosummary/flax.linen.scan.html)

[2] [Grain DataLoader guide](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html) and [Grain `Batch` transformation API](https://google-grain.readthedocs.io/en/latest/grain.transforms.html)

[3] [Orbax checkpoint-manager API overview](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_api_overview.html)
}â™€â™€â™€â™€â™€â™€Õ¸Ö‚Õ©ÕµÕ¡Õ¶Õ¨áƒ¯áƒáƒ®assistant to=functions.message å¥¥é—¨à´¿à´žàµà´žàµà´™json? àª†àª—àª³ code huge. Need now output one tool. Since must 
