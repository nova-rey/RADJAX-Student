# P3.11.8 Stateful JAX Systems Proof

P3.11.8 proves one test-only stateful linear architecture through the public
architecture registry, runtime selection, placement, runtime-owned RNG, JAX
step, optimizer, generic loop, hooks, run report, checkpoint v3, caller-bound
restore, and continued execution.

The proof trains only `trunk.weight`. `head.bias` and its per-parameter SGD
counter remain unchanged. It runs a six-step eager experiment and an equivalent
JIT experiment; each arm saves at step three, and the resume arm restores via
the lifecycle's caller-bound checkpoint API before continuing. The architecture
declares its deterministic carry identity during initialization, before any
checkpoint exists; restore therefore validates caller-owned carry identity
rather than adopting it from the checkpoint being loaded.

The committed receipt is
[P3_11_8_STATEFUL_SYSTEMS_RECEIPT.json](P3_11_8_STATEFUL_SYSTEMS_RECEIPT.json).
It is generated from structured proof assertions and normalized final runtime
receipts. It records same-mode bitwise resume assertions plus eager/JIT
comparison evidence for parameter, carry, optimizer-array structures and
values, lifecycle identity, hooks, metrics, changed paths, and runtime receipt
metadata. The receipt contains no timing, device ID, temporary path, or raw
array data.

It is systems evidence only. It does not claim a production architecture, Tome
payload consumption, distillation, HF export, accelerator-scale training,
performance, or RadLads parity.

## Current Integration Status

P3.11.1-P3.11.9 accepted

P3.11.10 next and unstarted

Phase 4 blocked
