# P4.1 Architecture Ingestion Contract Freeze

P4.1 freezes the ownership and evidence boundary for ingesting the first
production architecture plugin. It introduces no production RWKV package, no
generic-code change, and no compatibility claim beyond the one stated here.

## Baseline and inspected evidence

The checkpoint starts from `origin/main` commit
`773a0f5026f559ff7d9c1656cc4a09108436870c`. The tracked tree was clean before
P4.1; unrelated untracked files are preserved and never staged.

P4.1 also reconciles stale documentation that assigned `students/` removal to
this checkpoint. `students/` remains a deprecated compatibility namespace;
P4.1 neither removes nor refactors it. Any removal or migration requires a
separately scoped cleanup with explicit import, migration, deprecation, and
regression evidence.

The read-only inventory confirms these existing ownership seams:

| Concern | Existing authority |
| --- | --- |
| Explicit plugin registration and selection | `architecture.registry.ArchitectureRegistry` |
| Architecture contract and optional JAX capability | `architecture.protocols.ArchitecturePlugin` and `JaxArchitecturePlugin` |
| Configuration, init result, catalog/layout, carry, surface, HF evidence | Architecture plugin and `ArchitectureInitResult` |
| Production lifecycle construction | `learning.assemble_jax_learning_lifecycle` only |
| Generic step registration | `learning.composition` only |
| Callable declaration, identity, binding, and prepared identity | `runtime.callables` |
| Checkpoint/replay persistence | Existing generic v3 and replay owners |

`ArchitectureRegistry` has no discovery or default-registration policy.
`learning.composition` registers the generic learning callable, never an
architecture. No `src/radjax_student/architecture/rwkv7_reference/` package or
other RWKV production package existed at this baseline.

The direct dependency baseline is `numpy`, `PyYAML`, and the pinned
`radjax-contract` package; JAX is the declared optional `test-jax` dependency.
This checkpoint does not add a dependency.

## Frozen plugin shape and registration

P4.2 may create exactly this package shape:

```text
src/radjax_student/architecture/rwkv7_reference/
  __init__.py       # JAX-free static exports only
  config.py         # strict frozen reference configuration
  schema.py         # identity, layout, carry, surface, and HF projection
  plugin.py         # RWKV7ReferencePlugin
  kernels.py        # lazily imported pure-JAX execution kernels
  registration.py   # register_rwkv7_reference(registry)
```

The plugin ID is `radjax.architecture.rwkv7_reference`, version `1`.
Registration is explicit and caller-owned through
`register_rwkv7_reference(registry)`; neither registry discovery nor a default
registration in learning composition is allowed. The public package must not
import learning, runtime execution, optimizers, or validation.

## Ownership and capability progression

The plugin owns strict config validation, parameter catalog and layout,
persistent carry descriptor, parameter/carry initialization, objective-surface
resolution, and the authoritative HF descriptor plus derived reference. Runtime
owns backend selection, dispatch, key streams, and callable identity. Learning
assembles owner outputs without inspecting leaves. Objectives own objective
selection and execution. Checkpoint and replay owners remain architecture
neutral.

| Checkpoint | Permitted capability | Selection rule |
| --- | --- | --- |
| P4.2 | Static identity, config, schema, descriptor inspection | It must not advertise `architecture.jax_execution_v1`. |
| P4.3 | Deterministic architecture initialization | It remains non-executable. |
| P4.4 | Pure-JAX forward execution | It alone may implement and advertise the JAX capability. |

P3.12C already rejects a static-only plugin at the registry-selection boundary
because it is not a `JaxArchitecturePlugin`; it must reject before any missing
initializer or forward method is reached. Registry registration already rejects
a disagreement between declared and implemented JAX capability.

## RWKV authority, domain, and mapping rule

The mathematical authority is `BlinkDL/RWKV-LM` commit
`442120a5b40f7d764328bebde94324bc8790806f`, path
`RWKV-v7/rwkv_v7_numpy.py`. Its live-reviewed SHA-256 is
`dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6`.

The only frozen compatibility statement is exact step/sequence equation parity
with that pinned NumPy inference reference on a tiny float32 fixture domain:
vocabulary 16, hidden width 8, two blocks, head size 4, FFN width 16, and
context length 4. Persistent carry is exactly `last_x_time`,
`last_x_channel`, and `time_state_matrix`; token-local `v0` is not persisted.

P4.2's parameter mapping is a literal reviewed table for every equation-bearing
parameter. Every row must give the exact pinned logical name, RADJAX pytree
path, float32 shape rule, initialization claim, and direct/transformed/omitted
representation. A transformed or omitted parameter is permitted only with a
proof that it preserves the pinned equations; otherwise P4.2/P4.4 stops.
Source-compatible logical names do not claim PyTorch state-dictionary or
weight-file compatibility. Embedding and output head are untied.

This contract does not claim upstream initialization parity, optimizer grouping,
training-recipe parity, CUDA kernels, pretrained weights, weight-file loading,
model quality, cross-device bitwise equality, HF conversion,
`from_pretrained`, or `save_pretrained`.

## Generic-change ledger

No generic change is made in P4.1. The only pre-authorized later changes are:

| Change | Checkpoint | Owner | Architecture-neutral requirement |
| --- | --- | --- | --- |
| Sparse categorical cross-entropy objective | P4.5 | `objectives` | Any token-logit architecture can provide `[B,T,V]` logits and integer `[B,T]` token targets. |
| Initialization-key materializer | P4.3, only if required | `runtime.keys` | Any JAX architecture can consume a runtime-owned initialization reference without parsing seed identity. |

Any other generic change requires a recorded actual requirement, a non-RWKV
basis, an owner, a future non-RWKV use, and proof that the minimal alternative
is insufficient. Otherwise Phase 4 stops for human direction. Generic owners
must contain no RWKV branch, import, identifier, or policy.

## Advisory qrwkv-xla review log

`nova-rey/qrwkv-xla` was inspected only at advisory revision
`1adbb9bee92ed6ba68bc928d851b42cf80bf78eb`.

| Classification | Reviewed conclusion |
| --- | --- |
| Reusable lesson | Keep architecture math separate from runtime policy, and keep any optimized kernel from becoming the correctness authority. |
| Reusable lesson | Keep execution paths explicit and testable on small CPU-safe fixtures. |
| Rejected discrepancy | Its archived monorepo layout, historical student implementations, and teacher/Tome/Pallas scope are not imported, copied, or depended on. |
| Rejected discrepancy | Its behavior cannot supersede the pinned BlinkDL NumPy equations or broaden the declared fixture-domain claim. |

## P4.1 acceptance and exclusions

P4.1 passes only if a future architecture author can use the frozen package,
registration, ownership, capability, provenance, and mapping rules without
casually changing generic ownership. The next checkpoint is P4.2 static schema.

P4.1 excludes P4.2 implementation, JAX execution, parameter initialization,
teacher/Tome/distillation work, data, distributed or TPU work, Pallas, CLI or
serving, HF loading/export, pretrained weights, and a second architecture.
