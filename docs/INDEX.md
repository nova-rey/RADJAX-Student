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
- `architecture/`
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
Runtime work has begun under the locked roadmap; P2.3 backend registration and
selection are complete, and P2.4 CPU runtime execution is the next checkpoint.
