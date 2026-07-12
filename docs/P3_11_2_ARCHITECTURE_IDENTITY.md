# P3.11.2 Architecture Identity

`ArchitectureRegistry` is the sole architecture registry. It accepts only
objects that satisfy the complete `ArchitecturePlugin` contract.

JAX execution is an optional capability of that same plugin identity. A plugin
that declares `architecture.jax_execution_v1` must implement
`JaxArchitectureExecution`; a plugin that implements that execution protocol
must declare the capability. This rejects JAX-only objects and false
capability declarations without excluding non-JAX full-contract test doubles.

The former `radjax_student.students` protocol, registry, and tiny backend have
been removed. Explicit debug behavior remains under `radjax_student.debug`.
No Phase 4 module may restore a second architecture protocol or registry.
