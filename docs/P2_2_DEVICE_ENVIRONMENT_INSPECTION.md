# P2.2 Device and Environment Inspection

P2.2 implements architecture-independent observation of the local JAX execution
environment through:

```python
inspection = inspect_runtime_environment()
```

Inspection observes the machine. It does not select or control a backend.

## Result Model

`RuntimeInspection` contains:

- pass/fail status for inspection coherence;
- a P2.1 `RuntimeEnvironment`;
- a P2.1 `DeviceInventory` of normalized descriptors;
- structured warnings and blockers;
- explicit claims not made.

The result is immutable and serializes through `to_dict()` without raw JAX
objects, object identities, tracebacks, or temporary paths.

## Lazy JAX Boundary

Importing `radjax_student.runtime` remains safe without JAX. The inspection
module discovers and imports JAX/JAXLIB only inside
`inspect_runtime_environment()`.

JAX absence is a healthy observed fact:

```text
status: pass
jax_available: false
jax_version: null
jaxlib_version: null
platform/counts: null
warning: jax_not_installed
```

Installed-but-broken JAX is distinct and produces `jax_import_failed` with the
exception type/message but no traceback. JAXLIB discovery, import, and version
failures are reported separately while preserving any coherent JAX facts.

The base package still does not depend on JAX. The existing `jax` optional extra
remains the explicit installation boundary.

## Observed Facts

When public JAX APIs are available, inspection records:

- JAX and JAXLIB versions;
- default/active platform;
- process count and index;
- local and global device counts;
- distributed initialization state when publicly observable;
- visible global devices.

Requested `RuntimeConfig` policy is not an input to inspection and cannot be
rewritten as observed state. A requested TPU and an observed CPU remain two
different facts for P2.3 selection to compare later.

## Device Normalization

Every visible device becomes a `DeviceDescriptor` with a deterministic ID:

```text
<platform-or-unknown>:<process-or-unknown>:<inspection-index>
```

The descriptor may include public platform, kind, process index, local hardware
ID, and reported JAX device ID. It never retains the device object or its
`repr()`.

Per-device platforms are preserved. Multiple platforms produce
`heterogeneous_platforms_detected`; they are not collapsed into one claimed
platform.

Memory remains `None` because P2.2 does not rely on backend-private capacity
internals or vendor shell tools. Precision hints remain empty because device
visibility alone does not prove precision behavior. The corresponding
`device_memory_unknown` and `device_precision_unknown` warnings are honest
results, not inspection failures.

## Consistency and Failure

Counts are validated as nonnegative. When enough facts are known, inspection
checks process-index range, local/global count order, global inventory count,
per-device process range, and unique normalized IDs.

Missing public observations produce warnings and `None`. Contradictory or
un-normalizable facts produce `device_normalization_failed` blockers and a fail
status. An unexpected inspection implementation failure is converted to
`runtime_inspection_internal_error` rather than leaking a traceback.

Stable finding codes are exported as
`RUNTIME_INSPECTION_FINDING_CODES`.

## Doctor

`radjax-student doctor` now reports:

```text
Runtime Inspection
  status
  JAX/JAXLIB availability and versions
  observed platform
  process count/index
  global device count and kinds
  structured warning codes
  JAX execution: unavailable
```

Doctor stays healthy when optional JAX is absent. It fails only when runtime
inspection itself returns incoherent normalized data. JSON output includes the
complete `runtime_inspection` object.

## Non-Execution Boundary

P2.2 does not register/select a backend, compare requested capabilities, apply
fallback, initialize an execution context/runtime ID, create or place arrays,
call `jax.jit`, synchronize results, allocate a model, import architecture or
training modules, or use the network.

Run the focused gate:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_inspection.py
```

The matrix covers absent JAX, broken imports, JAXLIB failure, CPU/multi-device
normalization, heterogeneous platforms, unknown memory/precision, inconsistent
counts, deterministic JSON, doctor integration, and execution/network
sentinels. The complete Phase 1 and P2.1 gates remain required.

## Claim

P2.2 claims only:

```text
RADJAX-Student can inspect and normalize the local JAX execution
environment without selecting a backend or executing computation.
```

It does not claim backend registration/selection, requested-platform support,
fallback, array creation/placement, compilation, synchronization, GPU/TPU or
distributed execution, precision behavior, measured memory capacity, finalized
RNG streams, runtime persistence, architecture support, payload loading, or
training.
