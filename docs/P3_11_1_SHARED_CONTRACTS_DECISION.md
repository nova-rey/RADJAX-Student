# P3.11.1 Shared Contracts Decision

`radjax_student.contracts` owns dependency-free stable parameter identities,
parameter-tree layouts, HF lifecycle references, and JAX optimizer-state
descriptors. Architecture, optimizer, learning, runtime, and HF may depend on
this package; it may not depend on any of them.

Existing public contract types remain at their established import paths while
later P3.11 checkpoints migrate their ownership or add compatibility re-exports.
This avoids a gratuitous serialized-schema migration while removing the need for
new lateral public-package imports.
