# P3.11.6 Runtime JAX Bridge

The runtime-to-JAX bridge is `runtime_jax_key_bridge.v1`. It hashes, in order:
schema version, root seed, stream name, global step, micro step, slot, and
invocation index with SHA-256. The first eight digest bytes become two canonical
big-endian `uint32` words and are wrapped as a JAX key only in the explicit
bridge module.

Declared slots are initialization, dropout, augmentation, architecture
stochastic state, optimizer stochastic state, evaluation, and runtime tests.
Raw JAX keys are execution values, never continuation-checkpoint values.

Runtime-owned preparation places parameter, architecture-carry, optimizer-state,
and batch pytrees together on the selected device. It validates the requested
precision policy before dispatch; architecture code receives values only and
does not inspect or select devices.
