# P3.11.4 JAX Optimizer Capability

JAX execution is an optional capability of the existing `OptimizerBackend`
identity. `SgdOptimizer` now declares `optimizer.jax_execution_v1`; no second
optimizer registry or standalone update implementation is introduced.

`JaxOptimizerState` carries three distinct concerns:

- `OptimizerState`: stable optimizer identity, logical paths, step, and
  serializable metadata;
- `JaxOptimizerStateDescriptor`: capability, optimizer-state schema, canonical
  array keypaths, and parameter-layout digest;
- `arrays`: algorithm-owned numerical pytree that learning transports but does
  not inspect.

The pure update returns updated parameters, numerical optimizer state, and
metrics. It consumes only the layout-derived update mask, honors resolved
learning-rate schedule values, preserves excluded state leaves, and reports a
finite-gradient signal that the boundary turns into a stable optimizer error.
