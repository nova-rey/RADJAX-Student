# Phase 5 Execution Ledger

This append-only ledger records Phase 5 checkpoints. A checkpoint records its
own scope and verification but never its own commit SHA.

## P5.0A — Stateless Configurable RWKV-7 Instantiation

- **Status:** complete.
- **Changed-file summary:** configurable RWKV architecture-config validation,
  config-derived schema/layout/initialization, materialized-shape JAX token
  range, independent 64x5 proof, narrow documentation, and this ledger.
- **Tests and verification performed:** frozen Phase 4 schema, initialization,
  forward parity, lifecycle, checkpoint, receipt, and import-isolation tests;
  focused configurable 64x5 JAX proof; formatting and collection checks.
- **Generic-change decision:** none. The existing plugin remains one stateless
  registration at its current identity/version; no runtime, batch, objective,
  learning, artifact, checkpoint, or evaluation owner changed.
- **Unresolved non-blocking risks:** P5.3–P5.8 remain paused. This synthetic
  language contract provides no Tome compatibility or training semantics.
- **Next checkpoint:** none authorized by P5.0A.

## P5.1 — Behavior-Compilation Contract Freeze

- **Status:** complete.
- **Changed-file summary:** native-v3 Contract asset discovery/checksum
  verification and immutable admission projection; Contract pin; focused
  JAX-free tests; P5.1 contract freeze, index, roadmap, training-mode, this
  ledger, and `bible.md` updates.
- **Tests and verification performed:** Contract v4 asset closure and public
  Student admission tests; changed-file Ruff/format checks; a current Tome
  `native_v3_student_v4` package admitted through the Student API with
  `strict=True`.
- **Evidence or receipts changed:** the contract-freeze document records the
  Contract/Tome pins, static-asset checksum boundary, and local native-v3/v4
  identity probe. No generated Student receipt is committed.
- **Generic-change decision:** none. This adds a typed artifact boundary only;
  it does not change learning, objective, architecture, runtime, or checkpoint
  contracts.
- **Unresolved non-blocking risks:** P5.2 must prove hostile-resource opening,
  typed payload parsing, full passport preservation, and delivery-path
  semantic independence. No training behavior is accepted here.
- **Next checkpoint:** P5.2 — Validated Payload Access and Passport
  Preservation.

## P5.2 — Validated Payload Access and Passport Preservation

- **Status:** complete.
- **Changed-file summary:** JAX-free typed native-v3 payload decoder, public
  artifact exports, focused decoder tests, P5.2 contract documentation, this
  ledger, roadmap/status, and `bible.md`.
- **Tests and verification performed:** focused decoder/admission tests;
  current Tome `native_v3_student_v4` package admitted and decoded with strict
  Contract resource opens; Contract v4 adversarial/conformance suite and Tome
  strict v4 package checks.
- **Evidence or receipts changed:** P5.2 documentation records the sole
  Contract verified-opener path, declared resource metadata retained, and the
  JAX-free/no-batching boundary. No generated artifact receipt is committed.
- **Generic-change decision:** none. Artifact parsing remains in the artifact
  layer and does not alter learning, objectives, architecture, runtime, or
  checkpoint behavior.
- **Unresolved non-blocking risks:** P5.3 must create explicit deterministic
  train/held-out splits and architecture-neutral materialized batches. P5.2
  does not claim fixture persistence, training execution, or model quality.
- **Next checkpoint:** P5.3 — Generic Behavioral Batch Materialization.

## P5.3 — Contract v6 Behavioral Authority Projection

- **Status:** complete.
- **Changed-file summary:** exact Contract v0.8.3 pin; strict public v6
  admission and neutral language projection; complete multipart target and
  corridor projection; verified JSON authority, JSONL/M7 record projection; focused
  adversarial checks; checkpoint documentation and `bible.md` evidence.
- **Tests and verification performed:** strict directory/archive admission and
  full public projection against the approved Tome fixture; Contract's focused
  adversarial v6 suite; Student projection unit/adversarial tests; changed-file
  lint/format checks.
- **Generic-change decision:** Contract remains the only archive, locator,
  resource-integrity, and behavioral-semantic owner. Student retains only
  immutable identity-bearing values and no physical delivery information, after
  closing the complete seven-resource behavioral-authority role set.
- **Unresolved non-blocking risks:** P5.4 owns split policy and neutral batch
  materialization. P5.3 creates neither.
- **Next checkpoint:** P5.4 — BehaviorSplitPolicyV1 and neutral batches.

## P5.4 — Freeze Split Policy and Materialize Neutral Batches

- **Status:** complete.
- **Changed-file summary:** cohesive immutable `behavior` models, versioned
  split policy, P5.3-projection materializer, adversarial unit evidence,
  checkpoint documentation, and this ledger.
- **Tests and verification performed:** literal four-example split proof;
  UTF-8 ordering and split-identity checks; duplicate/insufficient exemplar,
  malformed behavioral-source digest, incomplete join, invalid-coordinate,
  and invalid mode-statistic-interval rejection; committed directory/archive
  P5.3 projection integration that checks every corridor coordinate, passport,
  and selected payload remains leakage-free; lint and formatting checks.
- **Generic-change decision:** none. The materializer terminates the existing
  P5.3 public projection into architecture-neutral values and owns no Contract
  transport, objective, optimizer, learning pass, checkpoint, or evaluation.
- **Unresolved non-blocking risks:** P5.5 must separately define and qualify
  objective/SGD behavior. These batches make no learning-quality claim.
- **Next checkpoint:** P5.5 — independent objective gradients and SGD
  qualification.

## P5.5 — Student Objectives and SGD Qualification

- **Status:** complete.
- **Changed-file summary:** neutral behavioral objective policy and JAX loss
  functions; production qualification of the existing `sgd.v1` identity;
  independent objective/optimizer evidence; checkpoint documentation; and
  this ledger.
- **Tests and verification performed:** JIT and autodiff execution of corridor
  statistics and coarse sparse-tail cross entropy; exact aggregate-tail
  log-sum-exp proof; malformed declared-outcome rejection; independent
  corridor and exemplar qualification runs each proving scoped JAX updates,
  schedule use, NaN/Inf rejection before a candidate state escapes,
  state-envelope serialization, continuation-checkpoint replay, and
  deterministic gradients; existing optimizer regression coverage.
- **Generic-change decision:** none. The objective seam takes only P5.4 neutral
  values and logits. Existing SGD was qualified in place; no second optimizer,
  architecture assumption, training pass, checkpoint production, or evaluator
  was introduced.
- **Unresolved non-blocking risks:** P5.6 alone owns the training-partition
  corridor pass and its checkpoint. P5.5 makes no pass or quality claim.
- **Next checkpoint:** P5.6 — Corridor Training Pass and Checkpoint.

## P5.6 — Corridor Training Pass and Checkpoint

- **Status:** complete.
- **Changed-file summary:** architecture-neutral training-only corridor pass,
  identity-bound resumable checkpoint, focused adversarial/replay evidence,
  P5.6 boundary documentation, and this ledger.
- **Tests and verification performed:** JAX-enabled P5.4–P5.6 focused suite;
  corridor finite-loss/nonzero-gradient/parameter-update proof; canonical
  traversal and exact replay; held-out, changed-binding, and foreign optimizer
  state rejection. The checkpoint binds the validated optimizer envelope and
  JAX descriptor identity as well as Contract/Tome/receipt, language/HF,
  behavioral, architecture, split, ordering, batching, policy/reduction, and
  pass cursor identities.
- **Generic-change decision:** none. The pass takes a neutral batch and a
  caller-owned forward function; it has no artifact transport, tokenizer,
  passport, or architecture-specific behavior.
- **Unresolved non-blocking risks:** P5.7 alone owns exemplar continuation.
  P5.6 makes no exemplar, evaluation, package, quality, or readiness claim.
- **Next checkpoint:** P5.7 — Sequential Exemplar Pass and Final Checkpoint.

## P5.7 — Sequential Exemplar Pass and Final Checkpoint

- **Status:** complete.
- **Changed-file summary:** architecture-neutral exemplar-only continuation,
  final identity-bound checkpoint, adversarial/replay evidence, P5.7 boundary
  documentation, and this ledger.
- **Tests and verification performed:** JAX-enabled P5.6-to-P5.7 sequential
  execution; finite loss/nonzero gradient/parameter update; exact model,
  optimizer, and checkpoint replay; wrong predecessor, changed authority or
  architecture, held-out batch, reordered passports, and mixed objective
  rejection.
- **Generic-change decision:** none. The pass consumes a neutral batch plus a
  caller-owned forward function and P5.6 state; it has no archive, locator,
  tokenizer, or architecture-specific behavior.
- **Unresolved non-blocking risks:** P5.8 alone owns held-out evaluation. P5.7
  makes no evaluation, package, quality, or readiness claim.
- **Next checkpoint:** P5.8 — Held-out Evaluation.

## P5.8 — Held-out Evaluation

- **Status:** complete.
- **Changed-file summary:** architecture-neutral held-out evaluator and replayable
  report; adversarial JAX evidence; P5.8 boundary documentation, index, this
  ledger, and `bible.md`.
- **Tests and verification performed:** JAX-enabled P5.6-to-P5.8 execution;
  every held-out corridor coordinate and exemplar passport exactly once;
  separate finite deterministic metric reports; immutable final model and
  optimizer state; exact report replay; final/predecessor continuity, training
  leakage, incomplete cursor, and duplicate held-out evidence rejection.
- **Generic-change decision:** none. Evaluation consumes neutral batches, two
  immutable checkpoints, and a caller-owned forward function. It owns no
  optimizer, model update, artifact transport, tokenizer, locator, or
  architecture-specific behavior.
- **Unresolved non-blocking risks:** P5.8 is finite held-out behavioral
  evidence only; it makes no quality, generalization, package, or readiness
  claim.
- **Next checkpoint:** P5.9 — deterministic HF-shaped proof.

### P5.8 acceptance repair

- The evaluator now requires complete P5.4 `BehavioralBatchesV1` authority and
  binds its split/source plus exact held-out corridor-coordinate and exemplar-
  passport set identities. Partial, substituted, or duplicate held-out inputs
  fail closed before evaluation; focused JAX evidence covers partial corridor
  and exemplar submissions.

### P5.8 materialization-authority repair

- Completeness now originates at P5.4, not in caller-supplied P5.8 expected
  batches. `BehavioralMaterializationDescriptorV1` binds the policy/source,
  full split mapping, and all partition coordinate/passport sets; immutable
  `BehavioralBatchesV1` rejects a surface that no longer matches its descriptor.
  P5.8 records that descriptor identity before it accepts held-out evidence.

### P5.8 sealed-corridor-authority repair

- P5.4 now requires exact policy-selected exemplar passport sets in both
  partitions. P5.6 requires the sealed P5.4 materialization object at its
  public entry, validates its complete source/split authority and exact training
  corridor, then checkpoints the verified descriptor identity. P5.7/P5.8 only
  propagate that verified identity; free-form materialization strings are not
  accepted.

### P5.8 independent P5.3 passport-authority repair

- P5.6 now receives the admitted P5.3 behavioral projection alongside the P5.4
  materialization and derives the canonical selected-passport registry identity
  and set at that boundary. It rejects descriptor or submitted-partition
  disagreement before importing JAX. P5.7 and P5.8 bind and verify that registry
  identity through the checkpoint lineage. Focused adversarial evidence uses
  `dataclasses.replace` to forge a descriptor and held-out passport subset; the
  P5.6 public entry rejects it before execution while the ordinary P5.4–P5.8
  path remains covered.

### P5.8 P5.3 admission-attestation repair

- The native P5.3 factory now mints a module-private admission attestation for
  each projection only after strict Contract v6 admission and resource opening.
  P5.6 requires that exact minted instance, so direct construction,
  reconstruction, or `dataclasses.replace` cannot substitute content-equivalent
  projection values as an authority root. The attestation is process-local and
  retains no archive, locator, or delivery-path data.

## P5.9 — Deterministic HF-shaped Proof Package

- **Status:** complete.
- **Changed-file summary:** a narrow canonical proof-package boundary,
  deterministic package/replay/load tests, P5.9 documentation, index, this
  ledger, and `bible.md`.
- **Tests and verification performed:** factory-attested P5.3 through P5.8
  lineage binding; deterministic bytes and inventory hashes; exact replay and
  load; altered-byte, altered-lineage, and reconstructed-projection rejection.
- **Generic-change decision:** none. This produces evidence-only JSON from
  existing immutable values; it changes no artifact admission, model,
  tokenizer, optimizer, training, evaluation, HF export, or loading behavior.
- **Unresolved non-blocking risks:** the package is not a general HF export or
  pretrained loader, and does not claim quality, generalization, or Phase 6.
- **Next checkpoint:** none authorized by P5.9.

### P5.9 inventory and lineage-evidence repair

- The focused P5.9 tests now assert the literal nine-file inventory and every
  emitted JSON schema key. They independently mutate each major serialized
  provenance, identity, checkpoint, split, optimizer, evaluation, and policy
  field, requiring expected-identity loading or build-time lineage validation
  to fail. This is test-only evidence strengthening; package scope and bytes
  on ordinary input are unchanged.
