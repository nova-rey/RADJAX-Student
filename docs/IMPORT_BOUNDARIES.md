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

JAX is optional and deferred until the default scaffold is stable.

See [ARCHITECTURE_CHARTER.md](ARCHITECTURE_CHARTER.md) for the full dependency
direction rules.
