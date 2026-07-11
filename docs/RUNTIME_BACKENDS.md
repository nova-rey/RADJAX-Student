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

P2.2 provides lazy environment/device inspection through public JAX APIs when
JAX is installed. Inspection returns normalized P2.1 models and treats optional
JAX absence as a healthy fact; it does not select a backend or execute work.

P2.3 adds `RuntimeBackendRegistry`, serializable backend descriptors, and pure
selection over supplied P2.2 inspection facts. The default registry declares a
JAX backend without importing, initializing, or executing it; JAX can remain
registered but unavailable when it is not installed. Selection keeps registered,
available, capability-compatible, platform-compatible, and selected distinct.

An explicit platform requires a visible compatible device. `automatic` uses the
documented `gpu -> tpu -> metal -> cpu` order, while `unspecified` does not
silently choose an automatic target. Fallback is disallowed by default and is
always reported when explicitly allowed. Capability declarations remain
non-proof until an execution gate verifies them.

See [P2.3 Runtime Backend Registry and Selection](P2_3_RUNTIME_BACKEND_REGISTRY.md).
P2.4 proves one selected JAX CPU context: explicit `device_put`, eager pure
execution, explicit synchronization, host result validation, phase timing, and
teardown. It does not add JIT, sharding, replication, GPU/TPU, distributed
execution, model state, or training. See
[P2.4 Single-Device CPU Runtime Smoke](P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md).
Fast paths remain later optional layers and never become correctness paths.

P2.5 adds backend-neutral `RuntimeKeys` as the deterministic root-seed hierarchy
owned by `ExecutionContext`. It serializes semantic stream lineage rather than
JAX key objects and establishes no random model/training behavior. See
[P2.5 RNG and Reproducibility Contract](P2_5_RNG_AND_REPRODUCIBILITY.md).

P2.6 adds logical-axis and value-level placement declarations, with centralized
capability mapping and value -> plan -> runtime-config precedence. It does not
translate declarations into devices, meshes, or backend sharding objects. See
[P2.6 Placement and Sharding Intent](P2_6_PLACEMENT_AND_SHARDING_INTENT.md).

P2.7 centralizes pure eager/JIT preparation, dispatch, optional synchronization,
and diagnostic timing. Raw JAX compilation stays inside `JaxRuntimeBackend`;
generic callers handle only requests, opaque preparations, output metadata, and
structured results. See
[P2.7 Compilation and Execution Boundary](P2_7_COMPILATION_AND_EXECUTION_BOUNDARY.md).

P2.8 persists only a small runtime-owned envelope: identity, global step,
runtime config, root-seed lineage, environment/topology summaries, precision,
placement, and generic resume metadata. Canonical JSON plus a manifest, sizes,
and SHA-256 digests make restore portable and verifiable. Model parameters,
optimizer state, architecture data, compiled executables, raw devices, and raw
JAX keys are deliberately excluded. See
[P2.8 Runtime State Save/Restore](P2_8_RUNTIME_STATE_SAVE_RESTORE.md).

P2.9 runs the same selected-device runtime path for explicit CPU, GPU, and TPU
requests. It reuses selection, explicit placement, the P2.7 execution boundary,
synchronization, P2.8 state round-trip, and one target receipt shape. Missing
GPU/TPU hardware is an `unavailable` receipt, never a false pass. See
[P2.9 GPU/TPU Portability Smoke](P2_9_GPU_TPU_PORTABILITY_SMOKE.md).

P2.10 freezes these seams under one maintained acceptance gate. The gate runs a
deterministic shared CPU trace through inspection, selection, heartbeat,
execution, runtime-state validation, portability teardown receipt, and doctor
integration; GPU/TPU receipts remain external evidence when those targets exist.
See [P2.10 Runtime Acceptance Gate](P2_10_RUNTIME_ACCEPTANCE_GATE.md).
