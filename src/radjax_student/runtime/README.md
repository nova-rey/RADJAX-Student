# Runtime Contract

Runtime answers where and how execution occurs.

P2.1 defines immutable, serialization-ready runtime intent, environment/device
facts, capability declarations, compilation options, execution context,
runtime-owned state envelope, structured errors, backend protocol, and reports.

Generic modules use only the Python standard library and do not import JAX,
architecture, artifacts, training, schedules, or optional ML stacks. No backend
registry, environment inspection, device operation, compilation, state
persistence, or execution is implemented yet.

See [`docs/P2_1_RUNTIME_CONTRACT.md`](../../../docs/P2_1_RUNTIME_CONTRACT.md).
