# P2.10 Runtime Golden Acceptance Gate

**Status:** PASS

P2.10 formally closes Phase 2 without adding a runtime capability. The maintained
gate freezes the architecture-independent runtime pipeline and validates its
public seams together:

```text
RuntimeConfig
-> inspection
-> registry and selection
-> CPU heartbeat
-> deterministic RNG identity
-> placement intent
-> P2.7 execution
-> P2.8 state round-trip
-> P2.9 portability and teardown receipt
-> doctor integration
```

## Maintained Gate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance/runtime
```

The gate uses a deterministic fake JAX-shaped backend so default CI proves the
shared CPU lifecycle without making optional JAX, GPU, or TPU hardware a base
dependency. It exercises explicit selection and placement, the P2.4 heartbeat,
the P2.7 execution facade, P2.8 runtime-only state validation, P2.9 portability,
and final teardown timing/receipt behavior. Existing runtime unit suites remain
the detailed failure matrix behind this maintained acceptance surface.

`runtime_phase2_acceptance_receipt.json` is the committed machine-readable
closure artifact. It records the accepted P2.9.1 runtime commit, schema
versions, required CI commands, passing gates, optional external accelerator
receipt policy, phase status, and explicit non-claims. It contains no timestamp,
raw device, raw array, compiled handle, or model state.

## Accelerator Evidence

GPU and TPU remain optional external receipts. When available, run:

```bash
radjax-student doctor --portability-smoke gpu --format json
radjax-student doctor --portability-smoke tpu --format json
```

An absent accelerator is reported as `unavailable`, never as a pass. The P2.10
gate does not infer accelerator evidence from backend declarations.

## Phase Completion

P2.10 proves a verified, architecture-independent runtime can inspect, select,
execute tiny pure functions, preserve runtime identity, and use one shared path
across supported observed targets. It does not claim training, gradients,
optimizer behavior, model initialization, Tome payload loading, distributed
execution, sharding, scale, performance, or model quality.

```text
PHASE 2 - STUDENT RUNTIME COMPLETE
PHASE 3 - GENERIC TRAINING CORE UNBLOCKED
```
