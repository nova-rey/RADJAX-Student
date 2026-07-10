# Inspect and Doctor CLI

P1.9 exposes the Phase 1 artifact-understanding pipeline to humans and
automation. The commands are product surfaces over existing public APIs, not a
second artifact parser or compatibility implementation.

## Inspect

```bash
radjax-student inspect --tome /path/to/tome
```

The default human report includes normalized identity, validation facts,
behavioral surfaces, capability requirements, the declarative pass plan, the
compatibility verdict, stable blockers and warnings, and separate artifact and
Student claims not made.

Use JSON for automation:

```bash
radjax-student inspect --tome /path/to/tome --format json
```

The JSON report includes the complete artifact summary, provenance, validation
summary, validated content index, inferred defaults, selected profile, and
compatibility report. It contains no terminal styling.

Useful options are:

```text
--profile PROFILE_ID     Select a compatibility declaration.
--format human|json      Select the renderer; human is the default.
--output PATH            Write the selected format to a file.
--overwrite              Replace an existing output file explicitly.
--show-contents          Add the validated content index in human mode.
```

Parent directories for `--output` are created as needed. Existing files are
never replaced without `--overwrite`.

## Profiles

`metadata_inspection_only` is the only public profile in Phase 1. It declares
metadata inspection and no payload-consumption implementation. The accepted
production fixture therefore receives a compatibility `FAIL` with explicit
missing-capability and unevaluated-dimension blockers.

Tests can select the hidden `declaration_test_only` profile to prove pass-path
wiring. Its output is labeled test-only, and its declaration is not evidence of
implementation or execution readiness.

## Doctor

```bash
radjax-student doctor
radjax-student doctor --format json
```

Doctor reports package versions and commits when discoverable, checks required
Contract APIs, verifies the accepted canonical fixture digest, opens the fixture
through `open_tome_artifact()`, infers defaults, produces the expected honest
metadata-only compatibility failure, and verifies JSON serialization.

Doctor also runs P2.2 runtime inspection. Human and JSON output report optional
JAX/JAXLIB versions, observed platform/process/device facts, normalized device
kinds, and structured inspection warnings. Missing JAX is healthy and appears as
`jax_not_installed`; doctor still states that JAX execution is unavailable.

Doctor also previews P2.3 backend selection from the same inspection: registered
backend IDs, availability, declared platforms/capabilities, selected target when
one is eligible, and structured selection blockers or warnings. This preview
does not initialize a backend or execute JAX. An unavailable optional JAX
declaration does not make doctor unhealthy.

P2.4 CPU execution remains explicit:

```bash
radjax-student doctor --runtime-smoke
```

This runs the one eager JAX CPU heartbeat and includes its complete serialized
receipt in JSON output. The normal doctor command never executes it. A failed
requested smoke returns status `1` with structured receipt blockers; absent JAX
is a coherent failed smoke rather than a traceback.

Doctor also reports P2.6 placement declarations. `single_device_cpu_smoke_only`
is the sole concrete placement proof; replicated, data-sharded, model-sharded,
automatic, and unspecified declarations remain unresolved. The report never
creates a mesh, sharding object, or multi-device array.

Doctor also reports P2.7 execution-boundary availability without executing a
function. Eager and JIT are invoked only through explicit runtime APIs; automatic
resolves to eager with a warning, and normal doctor output remains non-executing.

An expected compatibility failure does not make doctor unhealthy. Missing or
changed fixtures, failed Contract imports, pipeline failures, or serialization
failures do. Runtime inspection fails doctor only when observation itself is
incoherent, not when optional JAX is absent.

## Exit Codes

```text
0  Command completed and compatibility/self-check status is pass.
1  Command completed and compatibility/self-check status is fail.
2  Artifact, Contract, profile, output, or CLI usage error.
3  Unexpected internal error.
```

Known artifact errors preserve Contract blocker codes and render without a
traceback. A compatibility exit of `1` is a valid product result.

## Current Boundary

The commands provide metadata inspection, run-default inference, and
compatibility reporting. They do not claim payload consumption, training,
checkpoint execution, architecture implementation, runtime execution, JAX
portability, Hugging Face export, or model quality. The legacy
`cli/train_student.py` module remains a deprecated tiny smoke shim and is not the
product `train` command.

P1.10 promotes the fixture, malformed variants, CLI behavior, reports, and
non-claims into the maintained [Phase 1 acceptance gate](P1_10_PHASE1_ACCEPTANCE_GATE.md).
Phase 1 is complete; these commands remain understanding surfaces, not evidence
of execution readiness.
