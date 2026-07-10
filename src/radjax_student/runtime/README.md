# Runtime Contract

Runtime answers where and how execution occurs.

P2.1 defines immutable, serialization-ready runtime intent, environment/device
facts, capability declarations, compilation options, execution context,
runtime-owned state envelope, structured errors, backend protocol, and reports.

Generic modules use only the Python standard library and do not import JAX,
architecture, artifacts, training, schedules, or optional ML stacks. No backend
registry, device operation, compilation, state persistence, or execution is
implemented yet.

P2.2 adds lazy environment/device inspection through
`inspect_runtime_environment()`. JAX absence is a healthy reported fact;
installed-but-broken imports, partial public facts, and normalization failures
remain distinguishable. Public results contain only P2.1 models, never raw JAX
devices.

See [`docs/P2_1_RUNTIME_CONTRACT.md`](../../../docs/P2_1_RUNTIME_CONTRACT.md).
See
[`docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md`](../../../docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md).
