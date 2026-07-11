# Runtime Contract

Runtime answers where and how execution occurs.

P2.1 defines immutable, serialization-ready runtime intent, environment/device
facts, capability declarations, compilation options, execution context,
runtime-owned state envelope, structured errors, backend protocol, and reports.

Generic modules use only the Python standard library and do not import JAX,
architecture, artifacts, training, schedules, or optional ML stacks. JAX-specific
work remains lazy and isolated in the JAX backend.

P2.2 adds lazy environment/device inspection through
`inspect_runtime_environment()`. JAX absence is a healthy reported fact;
installed-but-broken imports, partial public facts, and normalization failures
remain distinguishable. Public results contain only P2.1 models, never raw JAX
devices.

See [`docs/P2_1_RUNTIME_CONTRACT.md`](../../../docs/P2_1_RUNTIME_CONTRACT.md).
See
[`docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md`](../../../docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md).

P2.3 adds registration and selection, P2.4 proves one explicit CPU lifecycle,
P2.5 freezes runtime RNG identity, and P2.6 records portable placement intent.
P2.7 adds the single pure-function eager/JIT execution boundary with opaque
backend handles, explicit synchronization, static/donation policy, phase timing,
and structured reports. It does not implement model functions, gradients,
optimizers, training, distributed execution, or state persistence.

P2.8 finalizes a small versioned runtime-state envelope and stores it as
deterministic JSON plus a SHA-256 manifest. It persists runtime identity, step,
config, RNG lineage, environment/topology summaries, precision, placement, and
generic resume metadata. It deliberately excludes model/optimizer state,
compiled executables, raw devices, raw JAX keys, and architecture data. See
[`docs/P2_8_RUNTIME_STATE_SAVE_RESTORE.md`](../../../docs/P2_8_RUNTIME_STATE_SAVE_RESTORE.md).

P2.9 adds `run_portability_smoke(platform, mode)` for the one selected-device
CPU/GPU/TPU path. It reuses explicit selection and placement, P2.7 execution and
synchronization, then P2.8 runtime-state round-trip validation. GPU/TPU absence
is reported as `unavailable`; no architecture, mesh, sharding, or distributed
path is introduced. See
[`docs/P2_9_GPU_TPU_PORTABILITY_SMOKE.md`](../../../docs/P2_9_GPU_TPU_PORTABILITY_SMOKE.md).

See
[`docs/P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md`](../../../docs/P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md).
