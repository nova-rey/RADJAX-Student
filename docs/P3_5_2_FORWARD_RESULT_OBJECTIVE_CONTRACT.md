# P3.5.2 Forward-Result Objective Contract

P3.5.2 establishes the shared output boundary required before JAX learning
execution. `ArchitecturePlugin` remains the sole architecture identity and
`ArchitectureRegistry` remains the sole registry. The optional
`JaxArchitectureExecution` protocol is a capability of that plugin; it is not a
second architecture system.

`ForwardResult` now carries named runtime `surface_values` and exposes
`surface(surface_id)`. `final_output` is backed by the existing `outputs` field;
other objective surfaces are resolved from the architecture-owned mapping.
Runtime surface values are deliberately omitted from serialized contract
payloads, which remain metadata-only.

`ForwardObjectiveEvaluator` receives a selected surface, targets, weights, and
objective configuration. It never receives raw parameters. The existing scalar
objective path remains temporarily available while P3.5.3 proves the JAX path;
it must be quarantined behind an explicit legacy/debug adapter before P3.5.9.
