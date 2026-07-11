# P3.5.9 Regression And Import-Purity Closure

Base CI installs `.[dev]` and remains JAX-free. The dedicated `test-jax` extra
installs `jax[cpu]` and the JAX CI job runs the marked numerical tests. Fresh
subprocess checks import the root, architecture, and learning packages and
assert that optional ML stacks and the Tome producer are not loaded.

The static closure checks verify that runtime and architecture do not import
the legacy students namespace, the JAX correctness module contains no NumPy
or private `jax.jit` call, and production package roots do not re-export dense
target loaders or the tiny training smoke.
