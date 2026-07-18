# RADJAX-Student

RADJAX-Student is the foundation for a Hugging Face-aware, JAX/XLA-portable
student runtime that consumes validated RADJAX behavioral artifacts and compiles
teacher behavior into interchangeable student model architectures.

It does not load teacher models or run teacher inference. Teacher artifacts must
be produced externally by RADJAX-Tome.

Production Tome semantics are owned by RADJAX-Contract and versioned in the
[canonical Tome/Student consumer handoff](https://github.com/nova-rey/RADJAX-Contract/blob/main/docs/reference/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md).
Student opens that contract through `open_tome_artifact()` and exposes validated
identity, provenance, content references, behavioral surfaces, capability
requirements, and the declarative pass plan without loading training payloads.
`infer_run_defaults()` converts those facts into an immutable configuration seed
while leaving architecture, runtime, training budget, and later policy choices
unresolved.
`evaluate_student_compatibility()` compares that seed with an explicit Student
capability profile and returns a reproducible pass/fail report without executing
payload, model, runtime, checkpoint, or schedule behavior.

The Phase 1 pipeline is available from the command line:

```bash
radjax-student inspect --tome /path/to/tome
radjax-student doctor
```

`inspect` supports compact human output and complete deterministic JSON. Its
default `metadata_inspection_only` profile intentionally fails when a Tome
requires capabilities that Student has not implemented; that result is useful,
not a command error. `doctor` verifies the local Contract boundary, accepted
canonical fixture digest, and metadata-only reporting pipeline while listing
unavailable execution capabilities honestly. See the [CLI guide](docs/CLI.md)
for formats, file output, profiles, and exit codes.

The Contract Layer is closed by the maintained
[P1.10 production acceptance gate](docs/P1_10_PHASE1_ACCEPTANCE_GATE.md):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
```

That gate proves Student can normalize or explicitly reject the accepted
production Tome before model allocation. It does not claim payload consumption,
runtime execution, training, export, or model quality. Phase 2 Student Runtime
work is unblocked under that constraint.

Phase 2 follows the [locked runtime roadmap](docs/RADJAX_PHASE2_RUNTIME_ROADMAP.md).
[P2.1](docs/P2_1_RUNTIME_CONTRACT.md) defines the architecture-independent
runtime configuration, observation, capability, context, error, protocol,
state-envelope, and reporting contracts. It does not inspect devices, select a
backend, initialize JAX, or execute computation.

[P2.2](docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md) adds lazy local JAX
environment and device inspection. Missing JAX is reported coherently without
breaking the base install; observed devices normalize into stable JSON models.
Inspection does not select a backend or execute computation. P2.3 backend
registry and selection now compare requested policy with those supplied facts.
The default registry declares JAX without importing it, reports optional JAX
absence coherently, and never initializes or executes a backend. See
[P2.3 runtime backend registry and selection](docs/P2_3_RUNTIME_BACKEND_REGISTRY.md).

[P2.4](docs/P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md) proves the first execution
heartbeat: selected JAX CPU initialization, explicit placement of one tiny value,
eager pure execution, synchronization, result validation, timing, and teardown.
It is opt-in through `radjax-student doctor --runtime-smoke`; normal `doctor`
remains inspection and selection only.

[P2.5](docs/P2_5_RNG_AND_REPRODUCIBILITY.md) establishes one deterministic,
runtime-owned root-seed hierarchy with named immutable streams for future model,
data, dropout, augmentation, evaluation, and runtime-test behavior. It creates
no backend RNG object and does not implement stochastic model behavior.

[P2.6](docs/P2_6_PLACEMENT_AND_SHARDING_INTENT.md) adds portable logical axes,
value placement declarations, centralized capability mapping, and deterministic
precedence. It defines placement intent only: the CPU smoke is still the sole
concrete placement proof, with no mesh or multi-device sharding implementation.

[P2.7](docs/P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md) creates the one
architecture-independent pure-function boundary for eager and explicit JIT
execution, static/donation policy, synchronization, timing, and opaque reports.
It does not introduce model functions, gradients, or training.

[P2.8](docs/P2_8_RUNTIME_STATE_SAVE_RESTORE.md) persists a small versioned
runtime-owned envelope with deterministic JSON, manifest hashes, restore
validation, and explicit resume compatibility reporting. It stores no model,
optimizer, architecture, executable, raw device, or raw JAX key state. The
opt-in `radjax-student doctor --runtime-state-smoke` path uses a temporary
directory; normal doctor remains non-writing and non-executing.

[P2.9](docs/P2_9_GPU_TPU_PORTABILITY_SMOKE.md) provides one explicit
selected-device CPU/GPU/TPU smoke path. It reuses runtime selection, placement,
P2.7 execution, synchronization, and P2.8 metadata round-trip, returning a
target-specific receipt. `radjax-student doctor --portability-smoke gpu` and
`--portability-smoke tpu` report unavailable hardware honestly without claiming
an accelerator pass.

[P2.10](docs/P2_10_RUNTIME_ACCEPTANCE_GATE.md) closes Phase 2 with a maintained
runtime acceptance gate and the committed
[`runtime_phase2_acceptance_receipt.json`](runtime_phase2_acceptance_receipt.json).
The gate verifies the completed runtime pipeline without claiming training,
model behavior, distributed execution, sharding, performance, or quality. Phase
3 Generic Learning Core is active.

P3.1 now establishes the [Generic Learning Contract](docs/P3_1_GENERIC_LEARNING_CONTRACT.md):
immutable architecture-independent configuration, state, generic batch,
objective/update scope, metric, error, and reporting models. The scopes remain
independent, defaulting to whole-student updates and final-output objectives.
This is a contract checkpoint only: it does not invoke objectives, gradients,
optimizers, parameter updates, checkpoints, loops, or Tome loading. Phase 3
follows the [locked Generic Learning Core roadmap](docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md).

P3.2 establishes the [Student Architecture Plugin Contract](docs/P3_2_STUDENT_ARCHITECTURE_PLUGIN_CONTRACT.md).
Architecture plugins own stable parameter paths, named regions, optional
objective surfaces, batch compatibility, and the passive initialization/forward
socket. The generic core keeps update intent separate from objective intent;
runtime keeps execution policy. The included fake plugin is a non-numerical
contract test double, not a concrete Student model.

P3.3 establishes the [Optimizer Contract](docs/P3_3_OPTIMIZER_CONTRACT.md):
explicit optimizer configuration and opaque state, stable-path update requests,
and serializable update reports. The test-only scalar SGD backend proves
whole-student and partial update masking while keeping excluded parameter values
and per-parameter state unchanged. It is not an Optax integration or a learning
loop.

## Current Integration Status

P3.11.1-P3.11.10 locally accepted

P3.11 integration closure complete

P3.12A locally accepted

P3.12A.1 removes the deprecated split JAX objective authority from core production namespaces; registry-selected objective execution is the only active path.

P3.12B locally accepted

P3.12B.3 closes the typed JAX-free anti-cheat audit for the exact 22-positive,
77-adversarial boundary matrix; no permissive blocker matching or
gate-to-production dependency is permitted.

P3.12C locally accepted: one learning-owned production assembler now binds
registry-selected components into the executable JAX lifecycle.

P3.12D locally accepted: runtime binds the actual declared generic JAX-step
callable, its source-derived identity, and all compilation-relevant prepared
execution identity fields. P3.12 is closed; Phase 4 is next and unstarted.

The foundation audit closure keeps application callable registration in learning
composition, makes NumPy losses legacy-only, locks proof-owned validation
namespaces, and defines Phase 4 as architecture-plugin ingestion with RWKV only
as the first reference architecture. No Phase 4 implementation has begun.

Phase 4 remains unstarted

Phase 4 requires successful required remote base/JAX CI or an explicit repository-owner waiver

P3.4 establishes the [Generic Batch and Objective Contract](docs/P3_4_GENERIC_BATCH_AND_OBJECTIVE_CONTRACT.md):
behavior-neutral batch metadata, objective request/result models, and explicit
weighting policies. Future Tome adapters populate these models rather than
defining a parallel training vocabulary; this checkpoint does not load Tome data
or execute objectives.

P3.5 provides one [Single Learning Step](docs/P3_5_SINGLE_LEARNING_STEP.md): a
deterministic scalar contract proof that calls the architecture and optimizer
boundaries once, records metrics, advances state, and preserves excluded paths.
It is not an epoch, training loop, Tome objective, or model-quality claim.

P3.6 adds the [Model and Optimizer Checkpoint Contract](docs/P3_6_MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT.md):
separate architecture, learning, and optimizer components validated by a
deterministic SHA-256 manifest. It is a layered persistence contract, not a
distributed or production-scale checkpoint claim.

P3.9.1 amends that checkpoint boundary with `learning_checkpoint.v2`: resumable
batch-source state is stored in the integrity-covered `source.json` component,
not in an unsigned sidecar. The synthetic smoke now proves that a resumed run
matches uninterrupted execution across state, cadence, metrics, normalized hook
events, and the required report surfaces.

P3.7 adds the [Generic Learning Loop](docs/P3_7_GENERIC_LEARNING_LOOP.md), a
bounded orchestrator that delegates every update to P3.5 and retains generic
batch-source position for deterministic continuation.

P3.8B integrates observer-only hooks at generic loop lifecycle boundaries, and
P3.8C adds deterministic immutable reports only after the generic loop has
completed. Reports preserve lifecycle, warning, blocker, checkpoint, scope, and
bounded retained-metric evidence without controlling execution or emitting
external telemetry. P3.8D now closes the stack with a deterministic acceptance
receipt; P3.9 synthetic learning smoke is complete.

P3.9 now proves the complete generic learning machine on a deterministic
two-parameter `y = 2x + 1` synthetic problem. It routes MSE through the
accepted single-step, SGD, loop, metrics/hooks, P3.6 checkpoint, resume, and
P3.8C reporting seams while proving scoped update boundaries and exact replay.
Run `PYTHONPATH=src python3 -m radjax_student.learning.synthetic_smoke` for the
offline smoke receipt. This is systems evidence only, not a model-quality,
Tome-training, evaluation, or production-hyperparameter claim.

P3.10 closes the [Learning Core Golden Acceptance](docs/P3_10_LEARNING_CORE_GOLDEN_ACCEPTANCE.md)
with a deterministic public-API receipt covering P3.1 through P3.9. It audits
the completed contracts without adding execution behavior:

```bash
PYTHONPATH=src python3 -m radjax_student.learning.p3_10_acceptance
```

P3.5 adds a pure JAX linear learning contract through the Phase 2 runtime
execution boundary, while keeping the base install JAX-free. The dedicated
`test-jax` CI job proves eager/JIT agreement, autodiff, functional state, and
scoped updates. The NumPy implementation remains explicit legacy/debug only.

Maintained capability status:

```text
proven:
- production Tome metadata inspection
- compatibility evaluation
- runtime lifecycle and execution policy
- scalar generic learning orchestration
- checkpoint/resume mechanics
- pure JAX linear learning contract

not yet proven:
- real production architecture training
- Tome payload consumption
- behavioral distillation
- Hugging Face export
- model quality
- accelerator-scale training
```

P3.5.10 closes the architecture-integrity gate with a machine-readable receipt
at `docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json`. `radjax_student.students`
is a deprecated compatibility package only, architecturally dead by P3.5.9,
and scheduled for removal at the numbered P4.1 architecture implementation
checkpoint.

Start with the [documentation index](docs/INDEX.md) before extending the
codebase. The normative Phase 0 docs are:

- [Design philosophy](docs/DESIGN_PHILOSOPHY.md)
- [Development roadmap](docs/RADJAX_DEVELOPMENT_ROADMAP.md)
- [Architecture charter](docs/ARCHITECTURE_CHARTER.md)
- [Student split contract](docs/STUDENT_SPLIT_CONTRACT.md)
- [Production artifact view](docs/P1_6_STUDENT_ARTIFACT_VIEW.md)
- [Production run defaults](docs/P1_7_STUDENT_RUN_DEFAULTS.md)
- [Student compatibility report](docs/P1_8_STUDENT_COMPATIBILITY_REPORT.md)
- [Inspect and doctor CLI](docs/CLI.md)
- [Phase 1 production acceptance gate](docs/P1_10_PHASE1_ACCEPTANCE_GATE.md)
- [Phase 2 runtime roadmap](docs/RADJAX_PHASE2_RUNTIME_ROADMAP.md)
- [P2.1 runtime contract](docs/P2_1_RUNTIME_CONTRACT.md)
- [P2.2 device and environment inspection](docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md)
- [P2.3 runtime backend registry and selection](docs/P2_3_RUNTIME_BACKEND_REGISTRY.md)
- [P2.4 single-device CPU runtime smoke](docs/P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md)
- [P2.5 RNG and reproducibility contract](docs/P2_5_RNG_AND_REPRODUCIBILITY.md)
- [P2.6 placement and sharding intent](docs/P2_6_PLACEMENT_AND_SHARDING_INTENT.md)
- [P2.7 compilation and execution boundary](docs/P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md)
- [P2.8 runtime state save/restore](docs/P2_8_RUNTIME_STATE_SAVE_RESTORE.md)
- [P2.9 GPU/TPU portability smoke](docs/P2_9_GPU_TPU_PORTABILITY_SMOKE.md)
- [P2.10 runtime acceptance gate](docs/P2_10_RUNTIME_ACCEPTANCE_GATE.md)
- [Phase 3 Generic Learning roadmap](docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md)
- [P3.1 Generic Learning contract](docs/P3_1_GENERIC_LEARNING_CONTRACT.md)
- [P3.2 Student Architecture Plugin contract](docs/P3_2_STUDENT_ARCHITECTURE_PLUGIN_CONTRACT.md)
- [P3.3 Optimizer contract](docs/P3_3_OPTIMIZER_CONTRACT.md)
- [P3.4 Generic Batch and Objective contract](docs/P3_4_GENERIC_BATCH_AND_OBJECTIVE_CONTRACT.md)
- [P3.5 Single Learning Step](docs/P3_5_SINGLE_LEARNING_STEP.md)
- [P3.6 Model and Optimizer Checkpoint Contract](docs/P3_6_MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT.md)
- [P3.7 Generic Learning Loop](docs/P3_7_GENERIC_LEARNING_LOOP.md)

## Closure Scope

The current integration status above is local evidence only. Required remote
base/JAX CI remains an external Phase 4 prerequisite.

The closure makes no production architecture claim, no Tome payload consumption,
no distillation, no Hugging Face export, no accelerator-scale training, no
multi-device proof, no cross-hardware bitwise replay claim, no cross-version
bitwise replay claim, no performance claim, and no RadLads parity claim.
