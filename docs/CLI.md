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

An expected compatibility failure does not make doctor unhealthy. Missing or
changed fixtures, failed Contract imports, pipeline failures, or serialization
failures do.

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

P1.10 remains required to promote the production fixture, malformed variants,
CLI behavior, compatibility reports, and non-claims into the maintained golden
acceptance gate and formally close Phase 1.
