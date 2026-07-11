# Documentation Index

Start here for Phase 0 foundation context.

## Normative Phase 0 Documents

- [Design Philosophy](DESIGN_PHILOSOPHY.md): project compass and first
  principles.
- [Development Roadmap](RADJAX_DEVELOPMENT_ROADMAP.md): long-range capability
  map.
- [Architecture Charter](ARCHITECTURE_CHARTER.md): ownership, dependency
  direction, and module boundaries.
- [Student Split Contract](STUDENT_SPLIT_CONTRACT.md): practical product
  boundary, inputs, outputs, claims, and non-goals.

## Supporting Documents

- [Canonical Tome/Student Consumer Handoff](https://github.com/nova-rey/RADJAX-Contract/blob/main/docs/reference/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md):
  versioned production artifact semantics owned by RADJAX-Contract.
- [P1.3 Production Tome Gap Report](P1_3_PRODUCTION_TOME_GAP_REPORT.md):
  production contract comparison, gap ownership, and corrective recommendation.
- [P1.4 Production Tome Alignment Plan](P1_4_PRODUCTION_TOME_ALIGNMENT_PLAN.md):
  ordered cross-repository ownership, schema, fixture, and acceptance gates.
- [P1.6 Student Artifact View](P1_6_STUDENT_ARTIFACT_VIEW.md): production
  Contract-backed metadata view, legacy isolation, and current non-claims.
- [P1.7 Student Run Defaults](P1_7_STUDENT_RUN_DEFAULTS.md): source-separated
  artifact facts, capabilities, pass intent, user choices, and deferred policy.
- [P1.8 Student Compatibility Report](P1_8_STUDENT_COMPATIBILITY_REPORT.md):
  explicit profile-based readiness verdicts with structured blockers.
- [Inspect and Doctor CLI](CLI.md): Phase 1 human/JSON product surface,
  profiles, output handling, exit codes, and current non-capabilities.
- [P1.10 Phase 1 Acceptance Gate](P1_10_PHASE1_ACCEPTANCE_GATE.md): canonical
  fixture, malformed matrix, golden reports, receipt, guarantees, and
  non-claims that formally close the Contract Layer.
- [Locked Phase 2 Runtime Roadmap](RADJAX_PHASE2_RUNTIME_ROADMAP.md): ordered
  architecture-independent runtime checkpoints through the golden gate.
- [P2.1 Runtime Contract](P2_1_RUNTIME_CONTRACT.md): runtime ownership,
  terminology, typed models, capability/error vocabulary, backend protocol, and
  non-claims.
- [P2.2 Device and Environment Inspection](P2_2_DEVICE_ENVIRONMENT_INSPECTION.md):
  lazy JAX observation, normalized devices/topology, structured findings,
  doctor integration, and non-execution guarantees.
- [P2.3 Runtime Backend Registry and Selection](P2_3_RUNTIME_BACKEND_REGISTRY.md):
  declarative backend registry, deterministic policy selection, explicit
  fallback, doctor preview, and non-initialization guarantees.
- [P2.4 Single-Device CPU Runtime Smoke](P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md):
  opt-in JAX CPU initialization, explicit placement/execution/synchronization,
  receipt timing, cleanup, and constrained execution claims.
- [P2.5 RNG and Reproducibility Contract](P2_5_RNG_AND_REPRODUCIBILITY.md):
  one runtime-owned root seed, fixed named immutable stream lineage, and
  backend-neutral serialization.
- [P2.6 Placement and Sharding Intent](P2_6_PLACEMENT_AND_SHARDING_INTENT.md):
  topology-free logical axes, value placement declarations, validation,
  centralized capabilities, and unresolved-resolution boundary.
- [P2.7 Compilation and Execution Boundary](P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md):
  pure eager/JIT execution requests, opaque preparations, argument policy,
  synchronization, phase timing, and structured results.
- [P2.8 Runtime State Save/Restore](P2_8_RUNTIME_STATE_SAVE_RESTORE.md):
  versioned runtime-only state, deterministic JSON, manifests, integrity checks,
  restore validation, and explicit state smoke.
- [P2.9 GPU/TPU Portability Smoke](P2_9_GPU_TPU_PORTABILITY_SMOKE.md): one
  selected-device CPU/GPU/TPU path with explicit receipts and honest target
  availability.
- [P2.10 Runtime Acceptance Gate](P2_10_RUNTIME_ACCEPTANCE_GATE.md): maintained
  Phase 2 closure gate, machine-readable receipt, external accelerator evidence,
  and explicit runtime non-claims.
- [Locked Phase 3 Generic Learning Roadmap](RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md):
  ordered architecture-independent learning-core checkpoints.
- [P3.1 Generic Learning Contract](P3_1_GENERIC_LEARNING_CONTRACT.md): immutable
  learning vocabulary, independent objective/update scopes, serialization, and
  non-execution boundary.
- [P3.2 Student Architecture Plugin Contract](P3_2_STUDENT_ARCHITECTURE_PLUGIN_CONTRACT.md):
  architecture-owned parameter identity, scope and objective resolution, passive
  init/forward models, and test-plugin boundary.
- [P3.3 Optimizer Contract](P3_3_OPTIMIZER_CONTRACT.md): scoped update mechanics,
  opaque optimizer state, explicit policy vocabulary, and scalar SGD test proof.
- [P3.4 Generic Batch and Objective Contract](P3_4_GENERIC_BATCH_AND_OBJECTIVE_CONTRACT.md):
  behavior-neutral batches, objective request/result models, and weighting policy.
- [P3.5 Single Learning Step](P3_5_SINGLE_LEARNING_STEP.md): one deterministic
  scalar contract composition with targeted-update proof.
- [P3.6 Model and Optimizer Checkpoint Contract](P3_6_MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT.md):
  layered ownership, deterministic manifests, and SHA-256 validation.
- [P3.7 Generic Learning Loop](P3_7_GENERIC_LEARNING_LOOP.md): bounded generic
  batch consumption, exact stopping, and source-position continuation.
- [P3.8A Hook Lifecycle and Failure Policy](P3_8A_HOOK_LIFECYCLE_AND_FAILURE_POLICY.md):
  deterministic observer-only standalone hook dispatch.
- [P3.8B Learning Loop Hook Integration](P3_8B_LEARNING_LOOP_HOOK_INTEGRATION.md):
  lifecycle dispatch at generic loop boundaries.
- [P3.8 Metrics, Hooks, and Reporting](P3_8_METRICS_HOOKS_AND_REPORTING.md):
  observer-only learning-loop observability boundaries.
- [P3.8C Deterministic Run Reporting](P3_8C_DETERMINISTIC_RUN_REPORTING.md):
  immutable completed-run reports, bounded metric summaries, and opt-in
  post-completion attachment.
- [Architecture Overview](ARCHITECTURE.md)
- [Import Boundaries](IMPORT_BOUNDARIES.md)
- [Runtime Backends](RUNTIME_BACKENDS.md)
- [Student Backends](STUDENT_BACKENDS.md)
- [Training Modes](TRAINING_MODES.md)
- [Two-Cycle Experiment](TWO_CYCLE_EXPERIMENT.md)
- [Roadmap Checklist](ROADMAP.md)

## Repository Skeleton

Long-term implementation packages exist under `src/radjax_student/`:

- `artifacts/`
- `runtime/`
- `learning/`
- `architecture/`
- `optimizers/`
- `training/`
- `schedules/`
- `hf/`
- `reports/`
- `cli/`
- `validation/`

These packages are placement boundaries, not proof of implemented capability.
New work should land in its intended long-term package rather than expanding
transitional namespaces.

## Bible

`bible.md` is the project's institutional memory. It is intentionally
append-only.

Use it for:

- why important decisions were made
- lessons learned
- tradeoffs considered
- conversations that changed project direction
- historical context future contributors would otherwise lose

Do not use it for installation instructions, CLI documentation, API reference,
or ordinary module descriptions.

Phase 1 is complete under the maintained P1.10 acceptance gate. Phase 2 Student
Runtime is complete under the P2.10 runtime acceptance gate. P3.1 establishes
the Generic Learning Core contract, P3.2 establishes the architecture-plugin
contract, P3.3 establishes the optimizer contract, and P3.4 establishes generic
batch and objective vocabulary; P3.5 is next.
