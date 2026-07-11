# P3.5.3 Pure JAX Learning Through Runtime

P3.5.3 adds the optional `radjax_student.learning.jax_core` module. Importing
the base package, architecture contracts, or standard learning contracts does
not import JAX.

The pure loss composition accepts parameters, functional architecture state,
a `JaxBatch` pytree, static `JaxObjectiveConfig` metadata, and an explicit RNG
key. Architecture execution returns the shared `ForwardResult`; objectives
consume its selected surface and never receive parameters. Updated runtime
state is returned through auxiliary output and is not mutated or differentiated
at the batch boundary.

`build_value_and_grad_fn()` uses `jax.value_and_grad`. It does not select a
device or compile. Eager/JIT selection is sent through the accepted Phase 2
`JaxRuntimeBackend` execution boundary, which owns compilation, dispatch, and
synchronization.

The maintained JAX proof uses a two-parameter CPU linear architecture with
`y = 2x + 1`, verifies autodiff and scoped updates, and compares eager and
runtime-hosted JIT results.
