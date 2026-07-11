# P2.9 GPU/TPU Portability Smoke

P2.9 proves one architecture-independent, selected-device runtime path across
explicit CPU, GPU, and TPU target requests. There are no separate accelerator
product stacks.

## Public API

```python
from radjax_student.runtime import run_portability_smoke

receipt = run_portability_smoke("gpu", mode="eager")
```

Supported platforms are `cpu`, `gpu`, and `tpu`; modes are `eager` and explicit
`jit`. The receipt is immutable and JSON-serializable. It records target/device
identity, process and device counts, execution mode, placement policy, result
validation, synchronization, state round-trip, diagnostic timings, structured
blockers/warnings, and non-claims. It contains no raw devices, arrays, compiled
handles, or backend objects.

## Shared Path

Every target follows the same sequence:

```text
inspect
-> explicit platform selection
-> one local device
-> explicit placement of [1.0, 2.0, 3.0]
-> P2.7 x * 2 + 1 execution
-> synchronization and [3.0, 5.0, 7.0] validation
-> P2.8 runtime state save/load/compatibility
-> target receipt
```

Target facts may differ, but architecture-facing behavior does not. The smoke
selects one local device deterministically and creates no mesh, sharding,
replication, collectives, or distributed execution.

## Availability and Receipts

When GPU or TPU hardware is absent, the receipt has status `unavailable`; it is
not a pass and does not make normal doctor unhealthy. External accelerator runs
can emit the deterministic JSON receipt through doctor output, for example:

```bash
radjax-student doctor --portability-smoke gpu --format json
```

Use `--output` to retain an external receipt. The receipt includes JAX/JAXLIB
versions, target/device kind, topology counts, mode, result/state status, and
explicit non-claims, but no temporary absolute state path or timestamp.

## Claims

P2.9 claims the same small runtime smoke can run on supported observed CPU, GPU,
and TPU targets with explicit selection, placement, synchronization, validation,
runtime-state round-trip, and target-specific receipts. It does not claim
multi-device or distributed execution, sharding, training, gradients, optimizer
state, throughput, kernel quality, scale, model quality, or cross-target numeric
identity beyond the tiny smoke tolerance.
