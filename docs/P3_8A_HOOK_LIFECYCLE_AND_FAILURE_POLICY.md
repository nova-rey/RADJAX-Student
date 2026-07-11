# P3.8A Hook Lifecycle and Failure Policy

P3.8A defines standalone observer-only lifecycle hooks. Supported events are
`loop_start`, `batch_received`, `step_start`, `step_end`, `checkpoint`,
`loop_end`, and `failure`. Dispatch order is priority then stable hook ID.

Contexts contain immutable run metadata and metrics only. Hooks may return
generic metrics and warnings, but cannot receive or mutate parameters, gradients,
optimizer state, architecture state, runtime handles, scopes, or checkpoint data.

Failure policy is explicit: `fail_fast`, `warn_and_continue`, or `disable_hook`.
Failures preserve structured type/identity details, disabled IDs are returned in
stable order, and core blockers can be merged without being overwritten.

P3.8A does not integrate hooks into the loop. P3.8B, P3.8C, P3.8D, and P3.9
remain pending.
