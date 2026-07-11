# Student Backends

The legacy/debug scaffold provides `tiny_debug`, a NumPy-only backend for import,
registry, and one-step training tests.

Production QRWKV/RWKV backends will be migrated after Contract boundaries are
stable.

This namespace is transitional. The long-term boundary is architecture plugins:
plugins answer how a model computes, while runtime backends answer where and how
that computation executes. Future core API work should migrate toward an
`architecture/` package rather than expanding `students/` as the permanent
public surface. `students/` is now a deprecated compatibility package whose
removal is assigned to P4.1.
