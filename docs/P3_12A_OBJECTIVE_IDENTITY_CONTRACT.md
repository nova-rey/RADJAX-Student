# P3.12A Objective Identity Contract

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
`a01fc17c295f9b086a9f0af80f02e20fba32e3620d1929c7a931f94bedf088a7`.

The maintained non-claims are: no production architecture; no Tome payload consumption; no distillation; no Hugging Face export; no accelerator-scale training; no multi-device proof; no cross-hardware replay; no cross-version replay; no performance claim; and no RadLads-parity claim.

## Current Status

P3.11.1-P3.11.10 locally accepted

P3.11 integration closure complete

P3.12A locally accepted

P3.12B next and unstarted

Phase 4 remains unstarted

Phase 4 requires successful required remote base/JAX CI or an explicit
repository-owner waiver

Phase 4 requires successful required remote base/JAX CI or an explicit repository-owner waiver
