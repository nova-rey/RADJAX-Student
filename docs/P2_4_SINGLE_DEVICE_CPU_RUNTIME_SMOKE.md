# P2.4 Single-Device CPU Runtime Smoke

P2.4 is the first RADJAX-Student checkpoint allowed to execute computation. It
proves one deliberately small, architecture-independent JAX CPU path:

```text
inspect -> select JAX CPU -> initialize -> explicit placement
-> eager pure function -> synchronize -> validate -> close
```

It is a runtime heartbeat, not training, a model smoke, or a benchmark.

## Public API

```python
from radjax_student.runtime import run_single_device_cpu_smoke

receipt = run_single_device_cpu_smoke()
```

The default request is explicit: JAX backend, CPU platform, one device, eager
execution, one process, no fallback, and seed `0`. The function always runs the
existing P2.2 inspection and P2.3 selection seams before it can execute.

`CpuRuntimeSmokeReceipt` is immutable and JSON-serializable. It contains the
selected backend/platform/device, summarized inspection and selection evidence,
input/output shape and dtype metadata, phase timings, result/synchronization
flags, a P2.1 `RuntimeReport`, structured blockers/warnings, and non-claims.
It contains no raw JAX array or device object.

## CPU Execution Contract

P2.4 accepts only a selected `jax` CPU backend with:

```text
backend_id=jax
platform_preference=cpu
placement_policy=single_device
compilation_policy=eager
distributed_policy=disabled
fallback_policy=disallowed
process_count=1
```

One visible CPU descriptor is selected by stable device ID. When several CPUs
are visible, the first stable descriptor is reported with
`runtime_multiple_cpu_devices_first_selected`; no sharding or replication is
introduced.

The JAX backend lazily imports JAX only after this selection is approved. It
constructs a real `ExecutionContext`, explicitly uses `jax.device_put` for the
selected CPU device, evaluates `x * 2 + 1` eagerly for `[1.0, 2.0, 3.0]`, calls
the public `block_until_ready()` synchronization method, and validates the host
result `[3.0, 5.0, 7.0]` before reporting success.

The runtime ID is deterministic: `jax-cpu-smoke-seed-<seed>`. It is execution
metadata only, not model, optimizer, checkpoint, or persistent runtime state.

## Cleanup And Timings

The smoke records initialization, placement, execution dispatch,
synchronization, and teardown timings with monotonic high-resolution time.
Synchronization is timed independently, so completion is never inferred merely
from function return. These numbers are diagnostic observations, explicitly
marked `runtime_smoke_not_benchmark`, and must not be treated as performance
claims.

After context initialization, teardown is in `finally` and runs after placement,
execution, synchronization, or result-validation failure. A teardown failure is
reported as `runtime_teardown_failed` without hiding the original failure.

## Optional JAX And Doctor

JAX remains an optional extra:

```bash
python3 -m pip install -e '.[jax]'
```

Base imports remain safe without JAX. When absent, an explicit smoke produces a
structured failed receipt rather than a traceback.

Default doctor remains non-executing:

```bash
radjax-student doctor
```

Run the smoke explicitly through doctor:

```bash
radjax-student doctor --runtime-smoke
radjax-student doctor --runtime-smoke --format json
```

The default doctor reports `NOT RUN`; `--runtime-smoke` embeds the complete
receipt and returns failure when the requested runtime heartbeat cannot pass.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_cpu_smoke.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
python3 -m ruff check src tests
```

Unit tests exercise successful execution via a deterministic fake CPU backend,
stable multi-CPU selection, invalid CPU/process/device policy, initialization,
placement, execution, synchronization, mismatch, teardown, JSON receipt, and
doctor integration. The real JAX CPU test runs when the optional JAX extra is
installed and otherwise skips cleanly.

## Claims Not Made

P2.4 proves a selected JAX CPU runtime can explicitly place a tiny value,
execute and synchronize one pure eager function, validate it, and close cleanly.
It does not prove JIT, GPU/TPU, distributed execution, sharding, replicated
placement, precision behavior, RNG stream consumption, runtime persistence,
model initialization, training, checkpoints, evaluation, export, or meaningful
performance.
