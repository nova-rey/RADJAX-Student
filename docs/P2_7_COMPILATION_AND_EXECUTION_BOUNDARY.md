# P2.7 Compilation And Execution Boundary

P2.7 centralizes preparation and execution of pure runtime functions. Callers
declare intent through serializable models; runtime backends own eager/JIT
realization, static/donation translation, synchronization, and opaque compiled
handles.

## Public API

```python
from radjax_student.runtime import (
    CompilationOptions,
    ExecutionRequest,
    execute_function,
)

request = ExecutionRequest(
    request_id="cpu-smoke-scale-v1",
    function_id="runtime.scale_add",
    mode="eager",
    compilation_options=CompilationOptions(
        mode="eager",
        synchronize_results=True,
    ),
)

output, result = execute_function(
    context=context,
    function=lambda x, scale: x * scale + 1,
    request=request,
    backend=backend,
    args=(2, 3),
)
```

`prepare_execution()` and `execute_prepared()` separate reusable preparation
from compilation, dispatch, and synchronization. `PreparedExecution` keeps its
backend handle and callable private; JSON includes only backend/function IDs,
mode, declared capabilities, timing, warnings, and preparation metadata.
`ExecutionResult` contains no raw output, only normalized metadata and phase
timings.

## Modes And Options

`CompilationOptions` finalizes the compact stable surface:

```text
mode
static_arg_names / static_arg_positions
donate_arg_names / donate_arg_positions
synchronize_results
debug
cache_policy
metadata
```

The historical `enabled` field remains a compatibility alias: `enabled=True`
normalizes to `mode="jit"`. New callers should use `mode`.

Modes are `eager`, `jit`, and `automatic`. Eager is the correctness baseline and
always reports compilation time as exactly zero. JIT is explicit and only the
JAX backend calls raw `jax.jit`, lowering, or compilation. P2.7 resolves
`automatic` to eager with `execution_automatic_mode_unresolved`; it never
silently enables JIT.

Static names/positions are validated against an inspectable callable signature.
Duplicate or negative positions and static/donation overlap are rejected. Donation
is opt-in, requires `execution.argument_donation_v1`, and records a warning that
memory effects are not proven.

## Capabilities And Timing

The centralized execution capability mapping is versioned as
`execution_capabilities.v1`:

```text
eager -> execution.eager_v1
jit   -> compilation.jit_v1
static arguments -> execution.static_arguments_v1
argument donation -> execution.argument_donation_v1
synchronization -> execution.synchronize_v1
```

Preparation, JIT compilation, dispatch, synchronization, and total timing are
separate. JIT compilation uses the backend's compilation phase before dispatch;
it is never labeled dispatch time. Timings are diagnostic and always carry
`execution_timings_not_benchmark`.

When `synchronize_results=False`, a successful report says dispatch occurred but
not that completion was observed, and emits `execution_not_synchronized`.

## JAX Isolation

Generic models and the execution facade import without JAX. The JAX backend
alone performs `jax.jit`, lowering/compilation, dispatch, and public completion
waiting. Importing `radjax_student.runtime` neither initializes JAX nor compiles
a function.

P2.4 remains the narrow CPU lifecycle acceptance path. P2.7 preserves that
explicit placement, synchronization, result validation, and teardown behavior;
it adds a reusable execution boundary rather than broadening the P2.4 claim.

## Doctor

Default `radjax-student doctor` remains non-executing. Its execution section
reports eager as available on explicit request, JIT as available only when JAX
is available, automatic as eager-with-warning, and default execution as not run.
The existing `--runtime-smoke` remains the deliberate P2.4 execution switch.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_execution.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_cpu_smoke.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
python3 -m ruff check src tests
```

Tests cover eager/JIT/automatic behavior, static and donation validation,
capability failures, each execution phase failure, synchronization reporting,
opaque JSON models, P2.4 compatibility, doctor reporting, import isolation, and
optional real JAX CPU eager/JIT paths.

## Claims Not Made

P2.7 claims one controlled boundary for pure eager and JIT function execution,
argument policy, synchronization, timing, and reporting. It does not claim
gradients, optimizer steps, training, compiled performance, persistent caches,
multi-device/GPU/TPU execution, sharding, distributed execution, architecture
model functions, Tome payload loading, or evaluation.
