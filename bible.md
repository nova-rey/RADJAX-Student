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

## 2026-07-10 - P1.6 production artifact-view correction

- Preserved the public `open_tome_artifact()` and `TomeArtifactView` seam while
  replacing its production dependence on one manifest, one payload, and one
  adapter with the accepted Contract cover-page, content-index, surface,
  capability, and pass-plan model.
- Student now normalizes one pinned production contract through published
  Contract APIs only. Contract blockers survive in the stable Student artifact
  error instead of being replaced by filename guesses or vague failures.
- Kept dense v0 behavior as an explicit legacy smoke/debug branch. Its manifest
  and payload fields are never production sources of truth.
- Corridor and exemplar convenience views expose metadata only. Student does not
  access assignment arrays or selected payload records, allocate a model, or
  execute training, runtimes, schedules, or checkpoints.
- Arbitrary unknown optional roles and surfaces remain inspectable, while
  required capabilities remain explicit for the later compatibility gate.

## 2026-07-10 - P1.7 production run-defaults correction

- Treated cover pages and behavioral surface declarations as run-configuration
  seed crystals: Student now derives identity, tokenizer, dimensions, surfaces,
  capabilities, and pass intent without asking users to repeat artifact facts.
- Kept user input limited to choices the Tome cannot know: student architecture,
  student size/config, training budget, and output location. Runtime, precision,
  optimizer, schedule, loss, evaluation, and export policy stay unresolved under
  their owning phases.
- Capability requirements are deterministic and visible before P1.8 evaluates
  compatibility. P1.7 does not infer readiness from a known surface kind.
- Preserved the artifact plan as immutable data with checkpoint boundaries and
  target scopes; no executable schedule, checkpoint, optimizer, or training
  behavior was introduced.
- Kept future surface kinds and plugin-defined target scopes extensible, while
  singular payload/adapter defaults remain isolated to legacy dense smoke use.

## 2026-07-10 - P1.8 Student compatibility report

- Separated Contract structural validity from Student readiness. A valid Tome
  can still fail because the declared Student profile lacks a surface schema,
  capability, target scope, tokenizer, dimension, or plan feature.
- Made compatibility capability-based with stable structured blocker codes.
  Missing requirements fail loudly instead of triggering hidden fallback or a
  vague not-compatible result.
- Added an honest metadata-inspection-only profile, which fails the production
  fixture because it declares no payload-consumption capabilities or dimension
  limits.
- Allowed a synthetic fully declared profile to pass evaluator logic while
  warning that declarations are not implementation or execution proof.
- Established P1.8 as the first reproducible yes/no gate before runtime action;
  no payload, model, architecture, runtime, checkpoint, or schedule execution
  was introduced.

## 2026-07-10 - P1.9 inspect and doctor CLI

- Made Phase 1 useful before training exists: humans and automation can now
  inspect a production Tome, see inferred defaults and required capabilities,
  receive an explicit compatibility verdict, and understand every blocker and
  non-claim.
- Kept `inspect` and `doctor` on the same Contract-backed artifact, defaults,
  and compatibility pipeline that later acceptance automation will depend on.
  The CLI owns only arguments, rendering, file output, and exit-code mapping.
- Treated failed compatibility as a legitimate product result rather than a
  malformed-artifact error. The public metadata-only profile fails honestly,
  while the hidden declaration-only test profile proves comparison wiring
  without pretending that declaration is implementation.
- Made doctor verify the accepted production fixture digest and report current
  capabilities and non-capabilities without loading Student training payloads,
  allocating a model, executing a schedule, or accessing the network.
- Left Phase 1 formally open. P1.10 must consolidate the golden fixture,
  malformed variants, CLI behavior, compatibility reports, and non-claims into
  one maintained acceptance gate.

## 2026-07-10 - P1.10 Phase 1 production acceptance gate

- Ended Phase 1 with proof instead of another feature. One maintained,
  offline, CPU-only gate now exercises the accepted Contract fixture through
  artifact opening, defaults, compatibility, inspect, and doctor.
- Made rejection first-class evidence. Student must preserve precise Contract
  blockers for malformed artifacts because accepting uncertain input is more
  dangerous than refusing valid-looking input with an explainable error.
- Froze normalized golden reports and a machine-readable receipt so fixture,
  dependency, blocker, warning, exit-code, and non-claim drift becomes visible
  in ordinary CI rather than relying on contributor memory.
- Established the honest Phase 1 guarantee: Student can understand or reject an
  accepted production Tome before model allocation. Contract may inspect
  indexed data structurally, but Student does not consume or expose production
  training payloads.
- Continued to refuse training claims. No model, architecture plugin, runtime,
  checkpoint, schedule, evaluation, export, quality, performance, scale, or
  parity behavior was proven by this gate.
- Unblocked Phase 2 because the artifact-understanding boundary is now
  reproducible and protected. Runtime work may begin only while preserving this
  acceptance gate and its explicit non-claims.

## 2026-07-10 - P2.1 runtime contract and terminology

- Separated runtime from architecture and training before adding JAX execution.
  Runtime owns where and how generic computation executes; architecture owns
  model math, and training/schedules own optimization mechanism and policy.
- Kept requested policy and observed environment as different immutable models.
  A GPU request is intent, not evidence that a GPU exists, and distributed
  policy is not observed distributed initialization.
- Versioned backend capability declarations so selection can reject missing
  semantics explicitly and future meaning changes cannot hide behind a reused
  boolean. Declarations remain non-proof until an execution gate tests them.
- Made fallback disallowed by default. Silently substituting CPU for a requested
  accelerator would change run meaning while making reports look successful;
  any compatible fallback must be requested and reported.
- Intentionally defined more than P2.1 executes. Stable config, environment,
  device, capability, context, compilation, state-envelope, error, protocol,
  and report concepts now give later checkpoints one coherent socket without
  claiming device inspection, backend selection, placement, JIT, persistence,
  architecture support, payload loading, or training.

## 2026-07-10 - P2.2 device and environment inspection

- Put observation before selection so P2.3 can compare requested policy with
  normalized machine facts instead of letting JAX defaults silently choose the
  meaning of a run.
- Treated JAX absence as a coherent fact rather than a crash. The base package
  remains useful and doctor remains healthy without the optional execution
  stack, while an installed-but-broken import stays distinguishable.
- Preserved unknown device memory, precision, distributed state, and topology
  as unknown. Guessing from device names or incomplete APIs would turn missing
  evidence into false capability claims.
- Used guarded public JAX APIs and normalized every device into immutable JSON
  facts. No raw device object, private backend internal, vendor shell command,
  object identity, or memory-address representation enters the public report.
- Kept inspection separate from execution proof. Seeing a CPU, GPU, or TPU does
  not prove placement, JIT, synchronization, precision behavior, distributed
  execution, runtime initialization, architecture support, or training.
