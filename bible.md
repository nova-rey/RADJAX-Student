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

## 2026-07-10 - P2.3 runtime backend registry and selection

- Made backend selection an explicit pure seam after observation and before
  initialization. Registry declarations report registration, availability,
  platform support, capability declarations, and selection evidence without
  retaining implementation objects or depending on architecture/training code.
- Kept JAX registered without importing or initializing it. Optional JAX absence
  is a coherent unavailable backend fact, not a crash and not a reason for the
  non-execution doctor to fail.
- Made platform policy visible: explicit targets require visible compatible
  devices, `automatic` has a deterministic accelerator-first order, and
  `unspecified` remains unresolved rather than silently using a JAX default.
- Preserved fallback as a semantic change that must be requested and reported.
  A GPU/TPU request cannot silently become CPU, and capability/policy blockers
  remain structured even where a compatible fallback exists.
- Treated capability declarations as selection inputs rather than proof. P2.3
  makes reproducible choices; P2.4 must still prove initialization, arrays,
  placement, execution, synchronization, and teardown on CPU.

## 2026-07-10 - P2.4 single-device CPU runtime smoke

- Started execution with one explicit CPU heartbeat because portable correctness
  needs a small, inspectable baseline before JIT, accelerator, sharding, or
  architecture-specific paths can be trusted. The smoke consumes the prior
  inspection and selection seams instead of letting a backend default decide.
- Required explicit placement and explicit synchronization. A returned JAX value
  is not sufficient evidence of target placement or completed work, so the
  receipt records the selected device, placed/output metadata, and completed
  synchronization before validating the deterministic result.
- Recorded phase timings as diagnostics, not benchmarks. The path is intentionally
  tiny and eager; its numbers explain where a smoke failed or waited, not runtime
  performance, model throughput, or accelerator quality.
- Tested teardown in `finally`, including failure paths. A cleanup error is
  preserved alongside the original initialization, placement, execution,
  synchronization, or validation blocker rather than rewriting execution history.
- Kept the claim narrow: P2.4 proves runtime heartbeat on selected JAX CPU only.
  It does not prove model initialization, Tome payload use, training, checkpoints,
  JIT, GPU/TPU, distributed execution, sharding, precision behavior, or speed.

## 2026-07-10 - P2.5 RNG and reproducibility contract

- Froze one root-seed contract before introducing model or training stochasticity.
  Runtime owns deterministic RNG identity so future subsystems cannot quietly
  create unrelated state and make a run unreproducible by construction.
- Made semantic stream names public and fixed: model initialization, data order,
  dropout, augmentation, evaluation, and runtime tests. Names and lineage are
  stable API, not implementation details that architecture or schedule code may
  redefine independently.
- Kept the tree backend-neutral and immutable. Serialized data contains only root
  seed, named lineage, deterministic derived seed, and metadata, never JAX/NumPy
  key objects or mutable generator state.
- Attached the tree to `ExecutionContext`, binding every initialized runtime to
  the exact seed hierarchy that it owns. Equal roots reproduce the tree; stream
  access cannot advance or alter another subsystem's randomness.
- Kept the claim honest: this is an RNG identity contract, not proof that model
  initialization, dropout, augmentation, training, evaluation, distributed RNG,
  or persistence behavior has run correctly.

## 2026-07-10 - P2.6 placement and sharding intent

- Defined logical axes as architecture-facing semantics without turning them into
  topology instructions. A plugin may say `batch` is data-oriented or `model` is
  model-oriented, while runtime remains the only owner of future mesh/device
  translation.
- Kept value paths stable and object-free so placement can be planned before
  parameter trees, Flax/Equinox structures, or architecture instances exist.
  This preserves the contract-first path from declaration to later backend work.
- Made `automatic` and `unspecified` materially different. Automatic delegates a
  future supported decision to runtime; unspecified intentionally makes no
  decision. Collapsing them would silently invent policy.
- Centralized placement-to-capability mapping so callers cannot infer support
  from an architecture name or scatter capability strings through model code.
  Contradictory declarations now fail as structured placement contract errors.
- Kept P2.6 declarative. It proves validated portable intent and a doctor report,
  not meshes, JAX sharding objects, replicated/data/model placement, topology
  solving, multi-device execution, distributed behavior, or training.

## 2026-07-10 - P2.7 compilation and execution boundary

- Centralized pure-function execution policy so architecture and training code
  cannot scatter raw JIT calls, donation switches, synchronization waits, and
  misleading timing around the repository. Callers declare intent; runtime owns
  backend realization and keeps compiled objects opaque.
- Kept eager as the correctness baseline and made automatic explicitly resolve
  to eager with a warning. JIT remains a requested capability, never an implicit
  optimization that changes semantics behind a caller's back.
- Split preparation, compilation, dispatch, and synchronization timing. First-use
  JIT compilation is measured before dispatch, while diagnostic timing remains
  clearly non-benchmark data rather than a performance claim.
- Made static arguments and donation constrained declarations. They validate
  signatures/capabilities and never activate automatically; donation records
  intent without claiming memory savings or backend cache behavior.
- Kept the claim narrow: P2.7 proves a runtime boundary for pure execution, not
  gradients, optimizer steps, training, model APIs, sharding, distributed work,
  acceleration, payload loading, evaluation, or compiled performance.

## 2026-07-10 - P2.8 runtime state save/restore foundation

- Made runtime persistence a small outer envelope rather than an early
  checkpoint format. Runtime owns identity, policy, step, RNG lineage, and
  historical environment/topology metadata; model and optimizer state require
  separate future contracts with their own ownership and validation rules.
- Used deterministic JSON, explicit schema versions, manifest sizes, and
  SHA-256 digests so a saved artifact is portable and independently verifiable.
  Restore refuses malformed, unsafe, incomplete, incompatible, or tampered
  state instead of trying to recover questionable runtime intent.
- Kept topology as historical metadata. It is useful for a resume comparison but
  does not recreate devices, migrate placement, or prove the current machine is
  execution-equivalent to the saved one.
- Revalidated the complete immutable RNG tree on load. A root seed alone is not
  sufficient evidence when derived stream lineage can be tampered with.
- Kept the P2.8 claim narrow: save/restore proves runtime metadata continuity,
  not model checkpoints, optimizer checkpoints, training resumption, distributed
  restoration, executable persistence, topology migration, or reproducibility of
  a complete run.

## 2026-07-10 - P2.9 GPU/TPU portability smoke

- Kept one shared selected-device path because three target-specific scripts can
  drift into three different products. Selection, placement, P2.7 execution,
  synchronization, result validation, and P2.8 state continuity now remain one
  runtime responsibility; only observed target facts and receipts vary.
- Treated unavailable accelerators as observed capability boundaries, not failed
  portability claims and not false passes. A GPU/TPU request without the target
  produces an explicit `unavailable` receipt so automation can distinguish lack
  of hardware from an executed regression.
- Required target receipts for external accelerator evidence. They carry stable
  target, topology, mode, validation, and non-claim data without raw backend
  objects, paths, timestamps, or benchmark implications.
- Kept timing diagnostic. Different accelerator compilation and dispatch times
  are environment observations, never throughput or kernel-quality benchmarks.
- Kept the claim narrow: P2.9 proves one small single-device runtime path on
  observed targets, not multi-device scale, sharding, distributed execution,
  training, gradients, optimizer behavior, or model quality.

## 2026-07-10 - P2.9.1 portability teardown receipt integrity

- Corrected the receipt lifecycle so cleanup is no longer outside the reported
  lifecycle. The portability outcome is now provisional until teardown completes,
  its duration is recorded, and only then is the immutable receipt constructed.
- Made cleanup failure explicit and compositional. A teardown exception turns an
  otherwise passing receipt into a failure with
  `runtime_portability_teardown_failed`; when a phase already failed, the
  original blocker remains and teardown failure is appended deterministically.
- Kept this patch narrow. It changes neither target selection nor P2.7 execution
  or P2.8 state behavior; it makes the existing receipt truthful about the full
  runtime lifecycle.

## 2026-07-10 - P2.10 runtime golden acceptance gate

- Closed Phase 2 with a maintained acceptance gate instead of a new runtime
  capability. The gate composes the completed public seams into one deterministic
  CPU trace and makes the runtime foundation a dependency future training work
  can rely on without re-litigating every earlier phase.
- Committed a machine-readable receipt with the accepted P2.9.1 runtime commit,
  schema versions, gate inventory, required CI commands, external accelerator
  evidence policy, phase status, and explicit non-claims. It is intentionally
  timestamp-free and contains no raw devices, arrays, executables, or model data.
- Kept GPU and TPU evidence honest: optional target receipts remain external
  artifacts when hardware is observed. Backend declarations and absent hardware
  never become portability passes.
- Preserved the boundary at Phase completion. Passing P2.10 proves runtime
  infrastructure, not training, gradients, optimizer behavior, model
  initialization, Tome payload loading, distributed execution, sharding,
  performance, scale, or model quality.

## 2026-07-11 - P3.1 generic learning contract

- Named the Phase 3 foundation the Generic Learning Core because it defines how
  learning state and outcomes are represented without assigning architecture,
  Tome, or runtime meaning to them. This checkpoint deliberately defines more
  than it executes.
- Made targeted updates generic `UpdateScope` intent rather than hard-coded
  layers. Architecture plugins will own region interpretation and future
  resolution into stable parameter paths, keeping the core reusable across
  Student architectures.
- Kept `ObjectiveScope` separate from `UpdateScope`: where a signal is observed
  and which parameters may change are independent decisions. Whole-student
  updates and final-output objectives remain the boring public defaults.
- Reserved stable parameter and optimizer trees with deterministic update masks
  for future scoped updates. Removing and reinserting subtrees would make state
  continuity and reproducibility harder to reason about.
- Kept P3.1 standard-library-only and non-executing. It declares no gradients,
  optimizer calls, parameter updates, checkpoint files, loops, architecture
  plugins, or Tome loading, so later phases must add each capability explicitly.
