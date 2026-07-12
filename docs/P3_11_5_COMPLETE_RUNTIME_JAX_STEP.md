# P3.11.5 Complete Runtime-Hosted JAX Step

The production JAX step now performs one pure runtime-executed transition:

1. architecture JAX forward using the architecture-resolved objective scope;
2. objective evaluation and `jax.value_and_grad`;
3. optimizer-plugin JAX parameter and optimizer-array transition using the
   layout-derived mask;
4. functional architecture-carry transition.

The runtime receives the complete function and therefore its execution receipt
covers the update itself. The outer adapter only validates finite values and
reconstructs immutable `LearningStepResult`, `LearningState`, metric, logical
path, and opaque optimizer-state transport models.

`JaxBatchMaterializer` is an explicit boundary. Its sole P3.11 implementation
is a finite-JSON test materializer; it does not define a Tome payload contract.
The old partial JAX update lives only under `radjax_student.legacy`.
