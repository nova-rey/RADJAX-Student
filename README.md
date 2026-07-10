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
`infer_run_defaults()` converts those facts into an immutable configuration seed
while leaving architecture, runtime, training budget, and later policy choices
unresolved.
`evaluate_student_compatibility()` compares that seed with an explicit Student
capability profile and returns a reproducible pass/fail report without executing
payload, model, runtime, checkpoint, or schedule behavior.

The Phase 1 pipeline is available from the command line:

```bash
radjax-student inspect --tome /path/to/tome
radjax-student doctor
```

`inspect` supports compact human output and complete deterministic JSON. Its
default `metadata_inspection_only` profile intentionally fails when a Tome
requires capabilities that Student has not implemented; that result is useful,
not a command error. `doctor` verifies the local Contract boundary, accepted
canonical fixture digest, and metadata-only reporting pipeline while listing
unavailable execution capabilities honestly. See the [CLI guide](docs/CLI.md)
for formats, file output, profiles, and exit codes.

The Contract Layer is closed by the maintained
[P1.10 production acceptance gate](docs/P1_10_PHASE1_ACCEPTANCE_GATE.md):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
```

That gate proves Student can normalize or explicitly reject the accepted
production Tome before model allocation. It does not claim payload consumption,
runtime execution, training, export, or model quality. Phase 2 Student Runtime
work is unblocked under that constraint.

Phase 2 follows the [locked runtime roadmap](docs/RADJAX_PHASE2_RUNTIME_ROADMAP.md).
[P2.1](docs/P2_1_RUNTIME_CONTRACT.md) defines the architecture-independent
runtime configuration, observation, capability, context, error, protocol,
state-envelope, and reporting contracts. It does not inspect devices, select a
backend, initialize JAX, or execute computation.

[P2.2](docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md) adds lazy local JAX
environment and device inspection. Missing JAX is reported coherently without
breaking the base install; observed devices normalize into stable JSON models.
Inspection does not select a backend or execute computation. P2.3 backend
registry and selection now compare requested policy with those supplied facts.
The default registry declares JAX without importing it, reports optional JAX
absence coherently, and never initializes or executes a backend. See
[P2.3 runtime backend registry and selection](docs/P2_3_RUNTIME_BACKEND_REGISTRY.md).

[P2.4](docs/P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md) proves the first execution
heartbeat: selected JAX CPU initialization, explicit placement of one tiny value,
eager pure execution, synchronization, result validation, timing, and teardown.
It is opt-in through `radjax-student doctor --runtime-smoke`; normal `doctor`
remains inspection and selection only.

[P2.5](docs/P2_5_RNG_AND_REPRODUCIBILITY.md) establishes one deterministic,
runtime-owned root-seed hierarchy with named immutable streams for future model,
data, dropout, augmentation, evaluation, and runtime-test behavior. It creates
no backend RNG object and does not implement stochastic model behavior.

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
- [Production run defaults](docs/P1_7_STUDENT_RUN_DEFAULTS.md)
- [Student compatibility report](docs/P1_8_STUDENT_COMPATIBILITY_REPORT.md)
- [Inspect and doctor CLI](docs/CLI.md)
- [Phase 1 production acceptance gate](docs/P1_10_PHASE1_ACCEPTANCE_GATE.md)
- [Phase 2 runtime roadmap](docs/RADJAX_PHASE2_RUNTIME_ROADMAP.md)
- [P2.1 runtime contract](docs/P2_1_RUNTIME_CONTRACT.md)
- [P2.2 device and environment inspection](docs/P2_2_DEVICE_ENVIRONMENT_INSPECTION.md)
- [P2.3 runtime backend registry and selection](docs/P2_3_RUNTIME_BACKEND_REGISTRY.md)
- [P2.4 single-device CPU runtime smoke](docs/P2_4_SINGLE_DEVICE_CPU_RUNTIME_SMOKE.md)
- [P2.5 RNG and reproducibility contract](docs/P2_5_RNG_AND_REPRODUCIBILITY.md)
