# P3.5 Architecture Integrity And Cleanup

P3.5 audits and repairs the foundations delivered by Phases 0 through 3 before
Phase 4 architecture implementation begins. The phase is complete only when
architecture owns model math, objectives consume architecture outputs, the
JAX-native path is proven through runtime execution policy, transitional APIs
are architecturally dead, Hugging Face preservation is contract-backed,
checkpoint ownership is explicit, documentation is reconciled, and all prior
acceptance gates still pass.

The first checkpoint, P3.5.1, is audit-only. Its machine-readable inventory is
committed in [P3.5 dependency audit](P3_5_DEPENDENCY_AUDIT.json) and records
current blockers without changing runtime behavior.

The locked order is:

```text
P3.5.1  Repository boundary and dependency audit       AUDIT ONLY
P3.5.2  Forward-result objective contract               COMPLETE
P3.5.3  Pure JAX learning through runtime               COMPLETE
P3.5.4  Architecture namespace consolidation             IN PROGRESS
P3.5.5  Legacy and debug isolation                       COMPLETE
P3.5.6  Hugging Face preservation contract               COMPLETE
P3.5.7  Checkpoint ownership and migration               COMPLETE
P3.5.8  Documentation reconciliation                      PENDING
P3.5.9  Regression and import-purity closure              PENDING
P3.5.10 Final architecture-integrity golden gate          PENDING
```

No Phase 4 implementation may begin until P3.5.10 passes.
