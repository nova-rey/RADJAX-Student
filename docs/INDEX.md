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

- [P1.3 Production Tome Gap Report](P1_3_PRODUCTION_TOME_GAP_REPORT.md):
  production contract comparison, gap ownership, and corrective recommendation.
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

Phase 1 may begin from the question: how do we correctly consume a validated
Tome artifact?
