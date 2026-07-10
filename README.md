# RADJAX-Student

RADJAX-Student is the foundation for a Hugging Face-aware, JAX/XLA-portable
student runtime that consumes validated RADJAX behavioral artifacts and compiles
teacher behavior into interchangeable student model architectures.

It does not load teacher models or run teacher inference. Teacher artifacts must
be produced externally by RADJAX-Tome.

Production Tome semantics are owned by RADJAX-Contract and versioned in the
[canonical Tome/Student consumer handoff](https://github.com/nova-rey/RADJAX-Contract/blob/main/docs/reference/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md).
Student opens that contract through `open_tome_artifact()` and exposes validated
identity, provenance, content references, behavioral surfaces, capability
requirements, and the declarative pass plan without loading training payloads.

The initial scaffold uses NumPy for tiny debug smoke tests so default CI does
not require JAX, TPU, Pallas, torch, or transformers. It does not yet claim
working training, JAX portability, Hugging Face export, complete Tome
compatibility, or model quality.

Start with the [documentation index](docs/INDEX.md) before extending the
codebase. The normative Phase 0 docs are:

- [Design philosophy](docs/DESIGN_PHILOSOPHY.md)
- [Development roadmap](docs/RADJAX_DEVELOPMENT_ROADMAP.md)
- [Architecture charter](docs/ARCHITECTURE_CHARTER.md)
- [Student split contract](docs/STUDENT_SPLIT_CONTRACT.md)
- [Production artifact view](docs/P1_6_STUDENT_ARTIFACT_VIEW.md)
