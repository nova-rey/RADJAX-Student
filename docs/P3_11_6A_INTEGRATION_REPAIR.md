# P3.11.6A Integration Repair

P3.11.6A closes the integration defects found before checkpoint v3 work.

The neutral `radjax_student.contracts` package owns finite-JSON batches,
objective and update scopes, resolved selections, metrics, errors, parameter
layouts, and optimizer-state descriptors. Historical learning and architecture
imports are exact compatibility re-exports of those class objects.

The production JAX step requires complete architecture and optimizer plugin
identities. It validates the finite-JSON batch and materialized parameter and
optimizer trees before execution. Runtime derives an explicitly Threefry JAX
key from the named stream identity, places parameters, architecture carry,
optimizer arrays, and batch values under one precision policy, then executes
forward, objective, autodiff, optimizer updates, carry, and numeric counters.
The execution receipt records placement, precision, and RNG identity.

Changed and unchanged logical paths are derived from the optimizer's actual
changed-leaf tree. The pre-P3.11 handwritten gradient update is available only
through `radjax_student.legacy.jax_learning`.

P3.11.7 remains blocked until this repair and both base and JAX CI pass.
