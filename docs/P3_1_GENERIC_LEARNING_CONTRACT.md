# P3.1 Generic Learning Contract

P3.1 creates the architecture-independent vocabulary for learning without
executing learning. The public API is `radjax_student.learning`.

## Contract

The package defines immutable, finite-JSON models for:

- `LearningConfig` and `LearningState`;
- `LearningBatch` metadata envelopes;
- `UpdateScope`, `ObjectiveScope`, and `ResolvedUpdateSelection`;
- `CheckpointPolicy`, `MetricRecord`, `LossResult`, `LearningStepResult`, and
  `LearningReport`;
- structured `LearningIssue` and `LearningContractError` values.

`canonical_learning_json()` encodes a serialized model deterministically. The
models retain no model parameters, optimizer state, tensors, backend objects,
or checkpoint files.

## Scope Semantics

`UpdateScope` says which parameters may change. Its kinds are `whole_student`,
`named_region`, `parameter_paths`, and `plugin_defined`. The default is
`whole_student`.

`ObjectiveScope` says where a learning signal is observed. Its kinds are
`final_output`, `whole_student`, `named_region`, `intermediate_surface`, and
`plugin_defined`. The default is `final_output`.

The two concepts are independent. Region and target identifiers are opaque to
the learning core; a future architecture plugin owns their interpretation and
may resolve an update scope into stable parameter paths. Stable paths reserve a
deterministic update-mask approach rather than subtree removal and reinsertion.

## Boundaries

P3.1 imports only the Python standard library. It does not import JAX, Flax,
Equinox, Optax, architecture plugins, Tome code, or optional ML stacks.

It does not invoke an objective, resolve a region, compute gradients, invoke an
optimizer, update parameters, write checkpoints, run a learning loop, or load a
Tome. `ObjectiveEvaluator` and `UpdateScopeResolver` are passive protocol seams
for later phases.

## Claim

RADJAX-Student now has a stable, architecture-independent contract for learning
configuration, state, generic batches, scoped objectives and updates, metrics,
errors, and reporting. It does not yet claim that a Student can learn.
