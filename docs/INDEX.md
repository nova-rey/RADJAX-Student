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
- [P3.8D Observability Golden Acceptance Gate](P3_8D_OBSERVABILITY_GOLDEN_ACCEPTANCE_GATE.md):
  deterministic closure receipt for the complete P3.8 observability stack.
- [P3.9 Synthetic End-to-End Learning Smoke](P3_9_SYNTHETIC_END_TO_END_LEARNING_SMOKE.md):
  deterministic full-stack learning proof with scoped updates, integrity-covered
  checkpoint source state, resume equivalence, reporting, and replay evidence.
- [P3.10 Learning Core Golden Acceptance](P3_10_LEARNING_CORE_GOLDEN_ACCEPTANCE.md):
  final public-API audit of the completed Phase 3 learning core.
- [P3.11.7 Checkpoint v3 and Optimizer Step Identity](P3_11_7_CHECKPOINT_V3.md):
  optimizer-owned numerical-state validation, deterministic tensor sidecars,
  and continuation checkpoint step consistency.
- [P3.11.8 Stateful JAX Systems Proof](P3_11_8_STATEFUL_SYSTEMS_PROOF.md):
  public runtime-to-checkpoint conveyor proof with caller-bound resume.
- [P3.11.10 Final Adversarial Gate Addition](P3_11_10_FINAL_ADVERSARIAL_GATE.md):
  mandatory checkpoint tamper cases for optimizer identity and step integrity.
- [P3.5 Architecture Integrity Roadmap](P3_5_ARCHITECTURE_INTEGRITY_ROADMAP.md):
  ordered cleanup checkpoints required before Phase 4.
- [P3.5.2 Forward-Result Objective Contract](P3_5_2_FORWARD_RESULT_OBJECTIVE_CONTRACT.md)
- [P3.5.3 JAX-Native Learning](P3_5_3_JAX_NATIVE_LEARNING.md)
- [P3.5.4 Architecture Namespace Consolidation](P3_5_4_ARCHITECTURE_NAMESPACE_CONSOLIDATION.md)
- [P3.5.5 Legacy and Debug Isolation](P3_5_5_LEGACY_DEBUG_ISOLATION.md)
- [P3.5.6 HF Preservation Contract](P3_5_6_HF_PRESERVATION_CONTRACT.md)
- [P3.5.7 Checkpoint Ownership and Migration](P3_5_7_CHECKPOINT_OWNERSHIP_AND_MIGRATION.md)
- [P3.5.8 Documentation Reconciliation](P3_5_8_DOCUMENTATION_RECONCILIATION.md)
- [P3.5.10 Final Architecture Integrity Gate](P3_5_10_FINAL_ARCHITECTURE_INTEGRITY_GATE.md)
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
Runtime is complete under the P2.10 runtime acceptance gate. Phase 3 is closed
under the P3.10 learning-core gate. P3.5 architecture integrity is complete
under the P3.5.10 receipt; Phase 4 may begin from this boundary.
