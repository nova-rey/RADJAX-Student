# Phase 4 Execution Ledger

This append-only ledger records the factual completion state for P4.1 through
P4.8. A checkpoint's SHA is printed in its post-commit, post-push checkpoint
report rather than embedded in the same commit. A later checkpoint may cite an
already-known prior SHA.

## P4.1 — Architecture Ingestion Contract Freeze

- **Status:** complete
- **Changed files:** P4.1 contract; this ledger; README, roadmap, index, and
  `bible.md` status/route updates; documentation-only reconciliation of the
  deprecated `students/` compatibility namespace.
- **Tests and verification:** focused architecture, HF authority, production
  assembly, and runtime-callable identity tests; changed-file Ruff/format,
  compile, diff, and documentation-link checks; one primary and one
  verification audit.
- **Evidence or receipts changed:** read-only seam/dependency inventory,
  pinned-source SHA record, and advisory qrwkv-xla review log in the contract;
  no generated receipt changed.
- **Generic-change decision:** none; sparse CE and a possible runtime key
  materializer remain the only later pre-authorized changes.
- **Unresolved non-blocking risks:** P4.2 must add a subpackage-specific static
  import test, and P4.3 must keep all seed/key derivation runtime-owned.
- **Next checkpoint:** P4.2 — RWKV Reference Configuration and Static Schema.

## P4.2 — RWKV Reference Configuration and Static Schema

- **Status:** complete
- **Changed files:** static JAX-free RWKV reference config, schema, plugin, and
  explicit registration package; focused schema tests; parameter mapping;
  roadmap/index/status updates; this ledger; and `bible.md`.
- **Tests and verification:** 43 focused architecture, registry, HF authority,
  P3.12C assembly, and RWKV schema tests passed in the project JAX environment;
  changed-file Ruff, format, compile, diff, and import-isolation checks passed.
- **Evidence or receipts changed:** literal 72-row source-prefix mapping and
  checked-in parameter-order fixture specification; no generated receipt changed.
- **Generic-change decision:** none; P4.2 uses existing architecture and HF
  contracts, and adds no generic owner change.
- **Unresolved non-blocking risks:** P4.3 must initialize every declared leaf
  through runtime-owned key material only; P4.4 must prove the mapping against
  a deterministic pinned-source parity fixture.
- **Next checkpoint:** P4.3 — Parameter and Carry Initialization.

## P4.3 — Parameter and Carry Initialization

- **Status:** complete
- **Changed files:** runtime-owned initialization materializer; neutral
  `ArchitectureInitRequest` seam; learning assembly composition; RWKV
  initialization/carry implementation; bounded audit rule; focused proofs;
  P4.3 documentation; this ledger; and `bible.md`.
- **Tests and verification:** canonical-reference/materializer, complete
  initialization, import-isolation, P3.12C assembly, architecture contract,
  and P3.12C source-audit tests passed; changed-file Ruff, format, compile,
  and diff checks passed.
- **Evidence or receipts changed:** regenerated P3.5 dependency audit records
  the bounded concrete-plugin JAX allowance; P4.3 docs record initialization,
  carry, layout, and descriptor evidence without serializing key material.
- **Generic-change decision:** exactly three approved architecture-neutral
  changes: sparse categorical cross-entropy; runtime-owned initialization-key
  materializer; and runtime-supplied initialization material on
  `ArchitectureInitRequest`. No other generic change was made.
- **Unresolved non-blocking risks:** P4.4 must prove the pinned NumPy equations
  only on the declared tiny float32 fixture domain and keep `v0` token-local;
  P4.8 must reconcile the stale P3.11 recorded-gate closure wording before its
  final verification ladder.
- **Next checkpoint:** P4.4 — RWKV Recurrent Forward Kernel.

## P4.4 — RWKV Recurrent Forward Kernel

- **Status:** complete
- **Changed files:** lazy pure-JAX RWKV step/scan kernels and plugin execution
  boundary; bounded P3.12C audit allowance; independent oracle, fixture,
  provenance, focused tests, P4.4 documentation, status pages, this ledger,
  and `bible.md`.
- **Tests and verification:** independent NumPy-oracle and checked-in-fixture
  logits/carry parity, step/scan agreement, finite/carry-change, perturbation,
  malformed-input, capability, import-isolation, and deterministic-audit tests;
  changed-file Ruff, format, compile, diff, and source-boundary checks passed.
- **Evidence or receipts changed:** parity fixture/generator/oracle provenance
  and P3.5 dependency audit now record the lazy `kernels.py` JAX imports.
- **Generic-change decision:** none; the approved generic ledger remains
  exactly sparse CE, runtime initialization-key materialization, and the
  architecture-neutral runtime-supplied initialization-material request seam.
- **Unresolved non-blocking risks:** the parity claim is limited to the pinned
  tiny float32 source artifact; P4.5 must prove learning integration and its
  explicit cross-step carry gradient boundary; P4.8 retains the P3.11 wording
  reconciliation noted at P4.3.
- **Next checkpoint:** P4.5 — Objective Surface and Generic Learning Integration.

## P4.5 — Objective Surface and Generic Learning Integration

- **Status:** complete
- **Changed files:** generic sparse-CE objective/registry exports; RWKV batch
  validation and initialized-carry identity conformance; focused objective,
  initialization, and assembled lifecycle proofs; P4.5 documentation, status
  pages, this ledger, P3.5/P3.12C/P3.12D source-dependent receipts, and
  `bible.md`.
- **Tests and verification:** focused sparse-CE, RWKV initialization, and real
  eager/JIT P3.12C/P3.12D lifecycle tests prove finite loss/metrics/gradients,
  position-sensitive within-sequence gradients, parameter/optimizer/state/key/
  carry advance, later-step carry acceptance, zero cross-step carry gradients,
  and float32 eager/JIT agreement; changed-file Ruff, format, compile, diff,
  ownership, and base-import checks passed.
- **Evidence or receipts changed:** P4.5 lifecycle/gradient-boundary document,
  regenerated P3.5 dependency audit, and source-dependent P3.12C/D receipts;
  their frozen inventories are unchanged. No raw initialization key or new
  checkpoint/config/HF/report receipt serializes runtime key material.
- **Generic-change decision:** exactly the approved three changes remain:
  sparse categorical cross-entropy; runtime-owned initialization-key
  materializer; architecture-neutral runtime-supplied initialization material
  on `ArchitectureInitRequest`. No additional generic change was made.
- **Unresolved non-blocking risks:** P4.6 must prove generic checkpoint,
  restore, and replay identity after a real step; P4.8 retains the recorded
  P3.11/P3.12A documentation-status wording reconciliation. Full cross-step
  BPTT, truncated scheduling, and long-context recurrent training are not
  proven.
- **Next checkpoint:** P4.6 — Checkpoint, Restore, and Replay Identity.

## P4.6 — Checkpoint, Restore, and Replay Identity

- **Status:** complete
- **Changed files:** focused RWKV checkpoint/restore/replay proof and shared
  test-only lifecycle fixture; P4.6 documentation, status pages, this ledger,
  and `bible.md`. No production persistence format or generic source changed.
- **Tests and verification:** after a real sparse-CE step, generic v3 save,
  fresh lifecycle restore, direct logits/carry equality, and equal next generic
  step; current-owner configuration/layout/HF/carry mismatch rejection;
  existing v3/P3.12C/P3.12D regression, changed-file Ruff/format/compile/diff,
  and source-boundary checks.
- **Evidence or receipts changed:** P4.6 typed test/documentation evidence;
  no v3 format fork, receipt refresh, or raw key/callable serialization.
- **Generic-change decision:** none; the approved ledger remains exactly sparse
  categorical cross-entropy, runtime-owned initialization-key materialization,
  and architecture-neutral runtime-supplied initialization material.
- **Unresolved non-blocking risks:** callable identity is only compared where
  existing assembly/execution artifacts record it, not claimed checkpoint
  persisted; P4.8 retains recorded P3.11/P3.12A reconciliation, including the
  frozen P3.12A receipt's dependency/evidence digest mismatch against the
  Phase-4 surface, and the historical P3.5 acceptance runner's pre-P4.3
  all-JAX/Phase-4-unstarted assumptions. P4.6 changes no production source
  and does not cause that
  historical runner conflict; its public restore path has positive replay
  coverage while mutated-artifact rejection is exercised at the generic v3
  owner loader it delegates to.
- **Next checkpoint:** P4.7 — Architecture Ingestion Procedure and
  Anti-Contamination Proof.

## P4.7 — Architecture Ingestion Procedure and Anti-Contamination Proof

- **Status:** complete
- **Changed files:** bounded validation-owned architecture audit; focused
  ingestion-isolation test; future-plugin guide; status pages, this ledger, and
  `bible.md`. No architecture, runtime, learning, checkpoint, or objective
  behavior changed.
- **Tests and verification:** current source passes the P4.7 literal audit;
  focused injected direct-import/identifier/validation/registration blockers
  pass; P3.5 audit regression, changed-file Ruff/format/compile/diff, and
  P3.12C/P3.12D recorded checks pass.
- **Evidence or receipts changed:** P4.7 guide and compact audit report record
  the bounded scope, checklist, and approved generic-change justifications;
  the clean P3.5 audit report is byte-stable, so no receipt refresh is claimed.
- **Generic-change decision:** none; the approved ledger remains exactly sparse
  categorical cross-entropy, runtime-owned initialization-key materialization,
  and architecture-neutral runtime-supplied initialization material.
- **Unresolved non-blocking risks:** the literal audit intentionally excludes
  aliases, taint/data flow, reflection, loaders, carrier analysis, and audit
  recursion; P4.8 retains P3.11/P3.12A reconciliation and the historical P3.5
  acceptance-runner conflict recorded by P4.6.
- **Next checkpoint:** P4.8 — Phase 4 End-to-End Acceptance.

## P4.8 — Phase 4 End-to-End Acceptance

- **Status:** complete
- **Changed files:** P4.8 validation runner/models/tests; the canonical Phase 4
  report and acceptance document; bounded P3.5/P3.11/P3.12/foundation status
  and receipt reconciliation; source-audit allowance; index/roadmap, ledger,
  and `bible.md` updates.
- **Tests and verification:** fresh-directory byte-identical P4.8 report,
  focused P4.8 tests, current P3.5/P3.11/P3.12A-D/foundation gates, final
  base/JAX suites, static checks, documentation/index checks, and two audits.
- **Evidence or receipts changed:** generated P4.8 report, P3.5 dependency
  audit, P3.11.10, P3.12A-D, and foundation receipts reflect current sources;
  no raw initialization key or callable value is serialized.
- **Generic-change decision:** none; exactly the three approved
  architecture-neutral changes remain sparse CE, runtime key materialization,
  and runtime-supplied `ArchitectureInitRequest` initialization material.
- **Unresolved non-blocking risks:** equation evidence remains the declared
  pinned tiny float32 fixture domain; remote CI remains unclaimed.
- **Next checkpoint:** none — Phase 4 is complete; do not begin P4.9 or Phase 5.
