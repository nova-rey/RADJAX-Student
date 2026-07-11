# Import Boundaries

RADJAX-Student may depend on RADJAX-Contract.

RADJAX-Contract owns schemas, validation, compatibility contracts, and artifact
definitions. RADJAX-Tome owns behavioral artifact production. RADJAX-Student
owns artifact consumption, runtime, training, architecture plugins, reports, and
Hugging Face export.

Student consumes Tome output through Contract-defined artifact files and
published Contract APIs. Student must not import Tome implementation modules.

Forbidden default imports:

- radjax_tome
- torch
- transformers
- datasets
- accelerate

JAX remains optional in the base install. `learning/jax_core.py` is the only
explicit JAX learning module; it is exercised by the dedicated `test-jax`
extra/job and is not imported by base package entrypoints.

The generic `learning/` contract package remains portable and passive. The
JAX adapter may receive parameters and architecture state, but objectives only
receive selected forward surfaces, targets, weights, and configuration. Runtime
owns JIT, device selection, placement, dispatch, and synchronization.

The `students/` package is a warning-emitting compatibility namespace only;
production code does not import it. Its removal is assigned to P4.1.

See [ARCHITECTURE_CHARTER.md](ARCHITECTURE_CHARTER.md) for the full dependency
direction rules.
