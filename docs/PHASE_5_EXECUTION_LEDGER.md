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
