# Optimizers

The optimizer boundary owns parameter-update mechanics over resolved stable
parameter paths. It does not own architecture meaning, runtime execution,
learning-step timing, schedules, checkpoints, or training loops.

P3.3 provides a scalar-mapping SGD test backend to prove scoped-update masking
and optimizer-state continuity. It is not an Optax adapter or a production
optimizer implementation.
