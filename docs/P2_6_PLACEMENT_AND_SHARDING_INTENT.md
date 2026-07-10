# P2.6 Placement And Sharding Intent

P2.6 freezes portable placement vocabulary before a runtime creates a mesh,
sharding object, or multi-device array. It records what callers intend, not how
JAX realizes it on a particular CPU, GPU, or TPU topology.

## Public Concepts

```python
from radjax_student.runtime import (
    LogicalAxisSpec,
    PlacementPlan,
    ValuePlacementSpec,
)

plan = PlacementPlan(
    plan_id="student-intent-v1",
    logical_axis_catalog=(
        LogicalAxisSpec("batch", sharding_role="data"),
        LogicalAxisSpec("model", sharding_role="model"),
    ),
    values=(
        ValuePlacementSpec("batch.tokens", "data_sharded", ("batch",)),
        ValuePlacementSpec("runtime.scalar_step", "single_device"),
    ),
)
```

`LogicalAxisSpec` describes a semantic plugin-facing axis with optional size,
role, required flag, and JSON metadata. Names are backend-neutral and
plugin-defined axes are preserved rather than forced into a closed architecture
vocabulary.

`ValuePlacementSpec` declares intent for a stable logical value path. It has no
parameter tree, Python object identity, or backend object. `PlacementPlan`
collects immutable value declarations, an axis catalog, a plan default, derived
capabilities, warnings, and non-claims. `PlacementResolution` reserves a
serializable result for later backend translation but remains unresolved in P2.6.

## Intent Vocabulary

```text
single_device
replicated
data_sharded
model_sharded
automatic
unspecified
```

The vocabulary is aligned with `RuntimeConfig.placement_policy`. Its centralized
versioned capability mapping is:

```text
single_device -> placement.single_device_v1
replicated    -> placement.replicated_v1
data_sharded  -> placement.data_sharded_v1
model_sharded -> placement.model_sharded_v1
automatic     -> no concrete capability yet
unspecified   -> no capability
```

`automatic` means runtime may choose a supported placement later.
`unspecified` means no placement choice exists yet. They remain distinct in
plans, precedence, warnings, and reserved resolutions.

## Validation And Precedence

Plans reject duplicate value paths and logical-axis names, unknown axis
references, invalid axis sizes, data/model declarations without matching logical
axis roles, replicated declarations carrying partitioned-axis constraints, and
unspecified values carrying concrete capabilities or constraints. Failures raise
`PlacementContractError`, which exposes a structured `RuntimeIssue` and stable
placement blocker code.

Effective placement uses declaration-only precedence:

```text
explicit value placement
-> plan default
-> RuntimeConfig.placement_policy
-> unspecified
```

This does not resolve devices, validate topology, or prove a concrete sharding
implementation. Constraints such as `must_be_divisible_by_device_count` are
preserved and marked unevaluated for later runtime policy.

## Doctor

`radjax-student doctor` reports all supported declarations, identifies the
single-device CPU smoke as the only concrete placement proof, and lists
replicated/data/model/automatic/unspecified as unresolved. Doctor does not
create a mesh or invoke placement execution because P2.6 remains declarative.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_placement.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
python3 -m ruff check src tests
```

Tests cover each intent, centralized capability mapping, duplicate and unknown
declarations, contradictory constraints, plugin axes, `automatic`/`unspecified`,
JSON round-trips, precedence, doctor reporting, and the absence of JAX,
topology, architecture, training, payload, and network dependencies.

## Claims Not Made

P2.6 does not claim replicated placement, data/model sharding, meshes,
`PartitionSpec`, `NamedSharding`, multi-device/distributed execution, JIT,
architecture parameter trees, model allocation, payload loading, or training.
It claims only portable, validated placement intent.
