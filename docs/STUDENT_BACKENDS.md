# Student Backends

The legacy/debug scaffold provides `tiny_debug`, a NumPy-only backend for import,
registry, and one-step training tests.

Production QRWKV/RWKV backends will be migrated after Contract boundaries are
stable.

This namespace is transitional. The long-term boundary is architecture plugins:
plugins answer how a model computes, while runtime backends answer where and how
that computation executes. The current production architecture contracts live
under `radjax_student.architecture`, and the RWKV-7 reference plugin belongs
there. `students/` is a deprecated compatibility namespace: no new Phase 4
architecture implementation may be added under it, although existing
compatibility code may remain temporarily. Its removal or migration requires a
separately scoped compatibility cleanup with an explicit import inventory,
migration plan, deprecation handling, and regression proof. That work is not
part of P4.1 or the current eight-checkpoint Phase 4 plan unless it directly
blocks plugin ingestion and human approval is obtained.
