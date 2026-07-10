# RADJAX-Student Bible

## 2026-07-09 - Dense Tome targets and sparse top-k loss

- Added a modular dense Tome target loader that validates Contract manifests,
  loads records, supports dense `.npy` shards, and exposes probability batches
  for training code.
- Added sparse top-k KL-style loss support for future compressed teacher target
  payloads while keeping current default tests NumPy-only.
- Configured pytest to include `src` on the import path so the suite runs from
  a fresh checkout without ad hoc environment variables.

## 2026-07-09 - P0.3 architecture charter

- Added the normative RADJAX design philosophy and development roadmap to repo
  docs so future work can be evaluated against the project compass.
- Added the Phase 0 architecture charter documenting dependency direction,
  plugin boundaries, runtime boundaries, target module layout, public product
  path, current implementation classification, and conflicts to resolve before
  further implementation.

## 2026-07-09 - P0.4 student split contract

- Added the Student split contract documenting RADJAX-Student's product
  boundary, expected inputs and outputs, happy path, non-goals, claims made,
  claims not made, upstream/downstream relationships, current-code
  classification, and Phase 1 entry criteria.
- Cleaned up Phase 0 Markdown formatting without changing normative content.

## 2026-07-09 - P0.5 repository skeleton lock

- Locked the repository skeleton to the Phase 0 charter shape with placeholder
  packages for runtime, architecture, schedules, Hugging Face integration,
  reports, and Student-side validation.
- Documented that `bible.md` is institutional memory, not API documentation or
  user-facing command reference, so future contributors know where decisions
  belong.
- Closed Foundation with the next Phase 1 question: how to correctly consume a
  validated Tome artifact.

## 2026-07-09 - P1.1 artifact view

- Added the first Phase 1 conveyor-belt station: a stable Tome artifact view
  that opens a Contract-valid Tome, exposes cover page and manifest metadata,
  and surfaces structural blockers instead of letting downstream code inspect
  raw on-disk layout.
- Kept the reader architecture-independent and runtime-independent. It infers
  defaults such as tokenizer identity, payload format, compression family, and
  expected adapter family without training, allocating models, or loading target
  tensors beyond Contract structural validation.

## 2026-07-09 - P1.2 inferred run defaults

- Added a run-defaults seed layer so future CLI and config resolution can start
  from what the Tome already knows instead of requiring users to repeat cover
  page metadata as flags.
- Treated cover pages as run-default seed crystals: user config should provide
  only choices the Tome cannot know, such as student architecture, student size,
  training budget, and output location.
- Kept unresolved runtime, optimizer, schedule, evaluation, and HF export
  choices explicit by phase so inferred defaults are not mistaken for a final
  training config.

## 2026-07-10 - P1.3 production Tome contract gap review

- Paused Phase 1 implementation to compare the initial Contract-backed Student
  view with the richer production Tome. The P1.1/P1.2 public seams remain
  useful, but the real artifact has distinct corridor and exemplar surfaces and
  cannot be represented truthfully by one manifest payload and one adapter.
- Modularity made the correction cheap: keep the artifact-opening and inferred
  defaults boundaries, then correct their inputs after RADJAX-Contract owns the
  production cover page, contents roles, surface schemas, capability set, and
  recommended pass order.
- Future work must follow the production semantic contract rather than the
  historical dense fixture shape. Modes and packed mode assignments are
  training-critical; fingerprints and delivery-path details are diagnostic.
- Recommended coordinated Contract and Student changes before any corridor or
  exemplar loader, loss, schedule, runtime, or training implementation.

## 2026-07-10 - P1.4 production Tome alignment plan

- Made P1.4 cross-repository because no one repository can repair an artifact
  boundary alone: Tome owns emitted truth, Contract owns shared meaning and
  validation, and Student owns normalized consumption.
- Forbid Student from compensating for incomplete producer indexing by walking
  directories, guessing filenames, importing Tome, or maintaining private
  schema knowledge. A failed upstream contract gate stops downstream work.
- Preserved the public `open_tome_artifact()` / `TomeArtifactView` and
  `infer_run_defaults()` seams while planning replacement of their provisional
  single-manifest, single-payload, and single-adapter internals.
- Required a generic surface collection, capability sets, surface-referenced
  pass recommendations, and extensible target scopes so corridor/exemplar are
  current typed projections rather than permanent limits on future research.

## 2026-07-10 - P1.5 upstream production alignment reference

- Linked Student documentation to the one canonical Contract-owned production
  Tome consumer handoff. Student does not copy producer schemas or fixtures.
- Kept P1.1/P1.2 correction parked while Tome completes emission and Contract
  completes shared validation. This repository change is documentation only;
  no artifact loader, run-default, training, or runtime behavior changed.
- P1.6 may begin only after the cross-repository P1.5 receipt records a passing
  shared production fixture gate.
