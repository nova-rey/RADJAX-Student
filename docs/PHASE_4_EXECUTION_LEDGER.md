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
