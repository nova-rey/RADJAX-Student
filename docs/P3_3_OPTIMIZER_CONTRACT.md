# P3.3 Optimizer Contract

P3.3 defines the architecture-independent optimizer boundary in
`radjax_student.optimizers`.

## Ownership

The optimizer owns parameter-update mechanics, optimizer-state shape, clipping
and weight-decay policy representation, and update reporting. Generic learning
will decide when a step occurs. Architecture plugins define parameter meaning;
runtime owns where computations execute.

The optimizer receives `ResolvedUpdateSelection`, never an architecture region
identifier. It may change selected stable paths while excluded paths remain in
the same parameter and optimizer-state trees.

## Stable Trees

`OptimizerState` records an optimizer ID, step, tracked paths, declared state
structure, and opaque backend-owned state. Serialization excludes raw tensors,
backend handles, Optax objects, and callables. The initial scoped-update policy
is explicit: excluded gradients are ignored, parameters remain unchanged, and
their per-parameter optimizer state does not advance.

`GradientTree` and update-result models provide a framework-neutral envelope for
opaque values plus serializable path and update metadata. `ParameterUpdate`
reports whether a path was applied, its update norm, and clipping status.

## Policies

`OptimizerConfig` makes learning rate, clipping mode/value, weight decay mode,
epsilon, momentum, and a schedule reference explicit. Clipping and weight decay
are never silent. P3.3 declares their vocabulary but does not claim every
backend implements every mode.

## SGD Test Backend

`SgdOptimizer` is a pure-Python scalar-mapping test backend. It initializes a
stable per-path step map, applies fixed or request-provided learning rates, and
proves whole-student and partial updates. It does not compute gradients, invoke
Optax, execute a full learning step, perform a training loop, or represent a
production performance path.

## Claim

RADJAX-Student has a stable architecture-independent optimizer contract for
configuration, state, scoped parameter updates, capability reporting, and
deterministic update metadata. It does not yet claim gradients, AdamW, schedules,
mixed precision, distributed optimization, checkpoints, Tome loading, or model
training.
