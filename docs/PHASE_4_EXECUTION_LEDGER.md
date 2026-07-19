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
