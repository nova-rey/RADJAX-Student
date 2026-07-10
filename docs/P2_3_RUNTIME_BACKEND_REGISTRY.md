# P2.3 Runtime Backend Registry and Selection

P2.3 adds an explicit, architecture-independent selection seam after P2.2
inspection and before P2.4 initialization. It answers which backend declarations
are registered, available in supplied observed facts, compatible with requested
policy, and selected. It does not initialize a backend or execute computation.

## Public API

```python
from radjax_student.runtime import (
    RuntimeBackendRegistry,
    build_default_runtime_registry,
    select_runtime_backend,
)

registry = RuntimeBackendRegistry()
registry.register(backend)

default_registry = build_default_runtime_registry()
selection = select_runtime_backend(config, inspection, default_registry)
```

Registries are instance-owned. There is no mutable global registry. Registration
is keyed by stable backend ID, rejects duplicates unless `replace=True`, and
lists declarations by stable backend ID rather than registration order.

`RuntimeBackendDescriptor` serializes the backend ID, implementation version,
supported platforms, versioned capability profile, observed availability, and
notes. It contains no implementation object or function reference.

## Initial Backend

`build_default_runtime_registry()` returns a new registry containing the `jax`
declaration. Construction neither imports JAX nor observes devices. The JAX
declaration is therefore registered even when JAX is absent; its availability is
then reported as unavailable from the supplied P2.2 inspection.

`FakeRuntimeBackend` is a deterministic declaration-only test helper. It is not
included in the production default registry.

P2.3 keeps five facts separate:

```text
registered -> available -> capability-compatible -> platform-compatible -> selected
```

None implies the next. In particular, seeing a device or declaring a capability
does not prove initialization, placement, compilation, synchronization,
precision behavior, or execution.

## Selection Rules

`select_runtime_backend()` accepts only a `RuntimeConfig`, a P2.2
`RuntimeInspection`, and a registry. It never reinspects the machine, imports
JAX, calls `device_put`, creates an array, initializes a context, compiles,
synchronizes, or runs a function.

An explicit backend ID limits eligibility to that declaration unless
`fallback_policy="allow_compatible"` is explicit. Missing IDs, unavailability,
platform mismatch, missing capabilities, and unsupported declared policies are
structured blockers. Each missing required capability gets its own
`runtime_capability_missing` finding.

When no backend is requested, eligible declarations are ranked deterministically:

```text
exact platform match -> no fallback -> stable backend ID
```

The stable-ID decision is reported with
`runtime_selection_used_tiebreak` whenever more than one declaration is
eligible.

Explicit platforms require a compatible visible device. `automatic` selects the
first compatible visible target in this fixed order:

```text
gpu -> tpu -> metal -> cpu
```

That inference is reported as `runtime_platform_inferred`. `unspecified` does
not become `automatic`: it leaves `selected_platform` unset and applies no
implicit target preference.

Fallback never silently substitutes CPU for GPU or TPU. It is used only when
the configured policy permits it, the selected declaration satisfies all
required capabilities, and the selected backend/platform is reported through
`fallback_used` plus `runtime_compatible_fallback_used`.

Distributed, placement, and JIT policies are checked against declared
capabilities. Explicit precision and distributed-auto behavior remain
unevaluated unless execution later proves them; their warnings do not erase
blockers.

## Doctor

`radjax-student doctor` now reports registered backend descriptors,
availability, supported platforms, declared capabilities, and a default
selection preview. This is informational and remains healthy when optional JAX
is absent. Doctor does not initialize a selected backend unless the explicit
P2.4 `--runtime-smoke` option is requested.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_registry.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_contract.py tests/test_runtime_inspection.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
python3 -m ruff check src tests
```

The P2.3 tests cover duplicate handling, deterministic listing, explicit and
fake backend selection, unavailable/missing-capability/platform blockers,
explicit fallback, automatic versus unspecified platform semantics, deterministic
tie-breaking, JSON serialization, doctor integration, and non-initialization.

## Claims Not Made

P2.3 claims an explicit, serializable, reproducible selection decision over
declared backend facts. It does not claim JAX initialization, arrays, placement,
JIT, synchronization, runtime execution, distributed execution, precision
correctness, architecture support, training, payload loading, export, or model
quality. P2.4 is the first execution checkpoint.
