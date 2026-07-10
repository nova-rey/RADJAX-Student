# Runtime Backends

P2.1 defines the architecture-independent `RuntimeBackend` protocol and shared
models. It intentionally provides no registry or concrete backend.

Runtime answers where and how execution occurs. It owns device policy,
precision policy, placement and compilation translation, synchronization,
runtime state, and execution reporting.

Runtime does not own architecture math, parameters, losses, optimizers, data
ordering, schedules, evaluation, or export. Generic runtime contracts import
only the Python standard library and remain safe to import without JAX device
initialization.

P2.2 adds environment inspection, P2.3 adds the registry and initial JAX backend
boundary, and P2.4 proves the first single-device CPU smoke. Fast paths remain
later optional layers and never become correctness paths.
