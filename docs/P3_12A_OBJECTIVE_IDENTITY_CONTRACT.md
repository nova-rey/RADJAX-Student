# P3.12A Objective Identity Contract

Current P3.12A evidence digest:
`af4af6f0d72d1e74c022107afd239ccfd36e1aaf932b0b7b153684b05787d993`.

P3.12A closes the hole where a caller could historically pair an arbitrary
objective implementation with a free-standing objective string. The canonical
contracts live under `radjax_student.contracts.objective`; the production
registry lives under `radjax_student.objectives`.

The canonical implementation is
`radjax.objective.mean_squared_error` version `1`. Its descriptor binds the
objective identity, capability-profile digest, objective-config digest,
architecture-owned resolved-surface identity, metric-schema identity, and
portable implementation identity. The MSE loss is separate from the emitted
`objective.mse` metric, so generic learning cannot relabel objective metrics.

The production JAX step accepts only a complete registry selection plus its
matching config and execution descriptor. Architecture owns surface resolution;
the objective receives that forward surface, targets, weights, and its own
config, never the parameter tree, optimizer state, runtime, or checkpoint.

Checkpoint v3 continuation restore is caller-bound to the expected objective
selection and descriptor. A v3 checkpoint missing this canonical block is
inspection-only and rejects normal continuation with
`checkpoint_objective_identity_missing`. Exact historical MSE aliases may be
reported as an explicit inspection migration, but are never silently treated as
a continuation-ready canonical objective.

The generated receipt is
[P3.12A objective identity evidence](P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json).
It comes from the accepted stateful eager/JIT conveyor, checkpoint restore,
replay verification, report preservation, and real registry/checkpoint/replay
adversaries. It does not claim a production architecture, Tome payload
consumption, distillation, Hugging Face export, model quality, multi-device or
accelerator-scale training, performance, or Phase 4 implementation.

The current executed evidence digest is
`af4af6f0d72d1e74c022107afd239ccfd36e1aaf932b0b7b153684b05787d993`.

## P3.12A.1 Authority Closure

P3.12A.1 removes the historical split objective authority from active core
production namespaces. `radjax_student.learning.jax_core` now exposes only
`build_registered_jax_loss_fn`; it cannot import, construct, or execute the
former evaluate-only objective/config pair. Historical JAX objective behavior
lives only in `radjax_student.legacy.objectives_jax`, emits a
`DeprecationWarning`, and cannot enter a modern lifecycle, JAX step, or
continuation restore. The architecture audit verifies the sole canonical
builder and rejects legacy exports, legacy configuration declarations, and
split arbitrary-objective signatures.

The maintained non-claims are: no production architecture; no Tome payload consumption; no distillation; no Hugging Face export; no accelerator-scale training; no multi-device proof; no cross-hardware replay; no cross-version replay; no performance claim; and no RadLads-parity claim.

## Current Status

P3.11.1-P3.11.10 locally accepted

P3.11 integration closure complete

P3.12A locally accepted

P3.12B locally accepted

P3.12C locally accepted

P3.12D locally accepted

Phase 4 architecture-plugin ingestion locally accepted

Phase 4 local acceptance does not claim remote CI success
# P3.12A - Objective Identity Contract

> Post-closure note: P3.12B now binds the HF lifecycle descriptor independently
> of objective identity. Objective selection and mathematics remain unchanged;
> P3.12C and P3.12D are locally accepted; P3.12 is closed and Phase 4
> architecture-plugin ingestion is locally accepted.

Current P3.12A receipt evidence digest:
`af4af6f0d72d1e74c022107afd239ccfd36e1aaf932b0b7b153684b05787d993`.
