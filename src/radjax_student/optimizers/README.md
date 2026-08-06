# Optimizers

The optimizer boundary owns parameter-update mechanics over resolved stable
parameter paths. It does not own architecture meaning, runtime execution,
learning-step timing, schedules, checkpoints, or training loops.

P5.5 production-qualifies the existing `sgd.v1` implementation for neutral
behavioral JAX objectives: it proves scoped-update masking, schedule use,
finite-gradient rejection, state continuity, serialization, and replay. It is
not an Optax adapter and does not imply momentum, clipping, weight decay,
distributed optimization, or any architecture-specific behavior.
