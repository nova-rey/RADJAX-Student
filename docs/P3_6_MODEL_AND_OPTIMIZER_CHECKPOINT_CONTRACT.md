# P3.6 Model and Optimizer Checkpoint Contract

P3.6 adds `radjax_student.checkpoints`, a deterministic layered persistence
contract. A learning checkpoint contains a runtime reference, learning progress,
architecture state and parameters, optimizer state, component manifest, and
SHA-256 integrity data.

Ownership mirrors the system: runtime identifies execution, learning owns
progress, architecture owns parameters and auxiliary state, and optimizer owns
its moments and tracked paths. Components are written separately and validated
against the manifest before restoration. Restore order is runtime, learning,
architecture, then optimizer.

This is a scalar-contract checkpoint proof. It does not claim distributed or
sharded checkpoints, Tome persistence, production checkpoint performance, or
execution equivalence.
