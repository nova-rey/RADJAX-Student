# P2.5 RNG And Reproducibility Contract

P2.5 freezes one runtime-owned deterministic randomness identity before model,
optimizer, or training code exists. The contract is intentionally backend-neutral
and never creates a JAX, NumPy, Python, or global mutable RNG object.

## Public API

```python
from radjax_student.runtime import RuntimeKeys

keys = RuntimeKeys.from_seed(1234)
```

`RuntimeKeys` is immutable and versioned as `runtime_keys.v1`. It derives these
public stream names in this fixed order:

```text
model_initialization
data_order
dropout
augmentation
evaluation
runtime_tests
```

Each `RuntimeKeyStream` contains only the root seed, semantic name, deterministic
lineage, derived integer seed, and JSON metadata. It contains no backend key or
backend object. Stream names and their order are part of the public contract.

## One Root Seed

`RuntimeConfig.seed` remains the only root-seed request. Every
`ExecutionContext` now materializes `RuntimeKeys.from_seed(root_seed)` and
serializes it as `runtime_keys`. A supplied tree must have the same root seed as
its context; mismatches are rejected.

Derivation uses a versioned SHA-256 namespace over the root seed and named
lineage. This makes equal roots reproduce the complete tree, different roots
diverge, and named streams remain isolated from one another. Accessing one stream
never mutates another stream or advances hidden global state.

## Ownership And Boundaries

Runtime owns RNG identity and lineage. Future architecture, data, and training
code must receive the appropriate named stream rather than construct independent
RNG state. P2.5 intentionally does not wire streams into model initialization,
dropout, augmentation, data loading, or evaluation behavior yet.

The serialized contract is portable before optional JAX is installed. A later
backend may derive its native key from the `derived_seed`, but backend key objects
must not enter `RuntimeKeys`, `ExecutionContext`, reports, receipts, or JSON.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_keys.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_contract.py
python3 -m ruff check src tests
```

Tests cover same-root reproduction, distinct-root divergence, fixed stream
ordering, named-stream isolation, serialization, execution-context attachment,
root mismatch rejection, and absence of backend/global RNG dependencies.

## Claims Not Made

P2.5 claims only one deterministic runtime-owned RNG contract. It does not
claim model initialization, dropout, augmentation, training, evaluation,
distributed RNG policy, backend RNG parity, RNG persistence, or random behavior
has been executed.
