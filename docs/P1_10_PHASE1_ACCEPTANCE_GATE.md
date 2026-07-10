# P1.10 Phase 1 Acceptance Gate

**Status:** PASS

**Gate version:** 1

**Generated:** 2026-07-10

P1.10 closes the Contract Layer by proving the behavior delivered in P1.6
through P1.9 as one maintained gate. It adds no Student capability.

## Accepted Inputs

| Input | Accepted identity |
| --- | --- |
| RADJAX-Tome producer | `fe5d51e769627cd89124fbb51dbdad2f80ad2fab` |
| RADJAX-Contract receipt | `ff8f6e9af976fc599ee31173d4f177fb1250b4d7` |
| RADJAX-Contract implementation | `cbce741f7c4c14f6716207e5838bf152cce73e49` |
| RADJAX-Contract package | `0.1.0` |
| RADJAX-Student accepted input | `eff271d7d09fbf30b8c45fcbf47dbcbe083091aa` |
| RADJAX-Student package | `0.1.0` |
| Fixture ID | `production_multi_surface_v1` |
| Fixture schema | `production_tome_fixture_v1` |
| Fixture digest | `468a259d518a28a6f60af8c339b124b65fd52da0640544d186eb9609933608d1` |

The Contract dependency remains pinned to the accepted receipt. A fixture,
schema, or dependency identity change requires a new upstream receipt and an
explicit gate update; tests never regenerate expected values automatically.

## Acceptance Matrix

The maintained tests under `tests/acceptance/` cover this complete path:

```text
Contract fixture
-> open_tome_artifact()
-> infer_run_defaults()
-> evaluate_student_compatibility()
-> inspect CLI
-> doctor CLI
```

The gate verifies:

- fixture-helper availability, fixture existence, exact tree digest, Contract
  package/pin identity, and direct Contract validation;
- production identity, provenance, producer/Contract status, claims, content
  index, arbitrary surfaces, corridor/exemplar metadata projections,
  capabilities, pass order, and checkpoint boundaries;
- no production `manifest.json` assumption, directory walk, or filename guess;
- deterministic defaults, unresolved user/policy fields, separate claims, and
  isolation of legacy singular payload/adapter values;
- honest metadata-only failure and controlled declaration-only pass, including
  all implemented identity, surface, capability, dimension, scope, plan,
  producer, and Contract blocker codes;
- Contract rejection and Student blocker preservation for path traversal,
  absolute paths, stale hashes/sizes, missing and duplicate roles/paths, invalid
  surface/pass references, bad mode linkage, mode/fingerprint confusion, packed
  array length mismatch, and malformed exemplar top-k, mask, token, and corridor
  linkage;
- unknown optional content remains inspectable and unknown required capability
  remains an explicit compatibility failure rather than a parse failure;
- human/JSON inspect output, healthy doctor semantics, all documented exit
  codes, both file formats, parent creation, overwrite refusal, and explicit
  overwrite;
- default-import isolation and sentinels for Student payload loading, model or
  architecture instantiation, runtime/schedule selection, checkpoint/training
  action, and network access;
- legacy dense-v0 smoke support remains available and cannot leak singular
  production assumptions into the accepted production path.

Contract structural validation may inspect indexed arrays and exemplar records
to establish artifact validity. Student does not load those through its target
loaders, expose them in `TomeArtifactView`, or consume them as training data.
This distinction is part of the gate and prevents a broader claim than the
actual architecture supports.

## Golden Reports

The canonical metadata-only inspect and doctor reports are checked in as:

```text
tests/golden/phase1_inspect_metadata_only.json
tests/golden/phase1_doctor.json
```

The inspect fixture path and doctor Python/package-discovery fields are
normalized. The receipt records each file's SHA-256 digest. No timestamps,
temporary paths, machine-specific package paths, or unordered values are
accepted into the goldens.

## Reproduce

The obvious CI gate is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/acceptance
```

Final verification also runs:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
python3 -m ruff check .
python3 -m ruff format --check .
```

The acceptance gate is deterministic, offline, CPU-only, independent of JAX
and optional ML packages, and fast enough to run on every CI change. Exact test
counts and results are recorded in `phase1_acceptance_receipt.json`.

## Result

Phase 1 now guarantees:

```text
Given a production Tome conforming to the accepted Contract,
Student can open it, normalize it, infer run facts, expose requirements,
produce an explicit compatibility verdict, render stable reports,
and reject malformed artifacts before model allocation.
```

Successful understanding and explicit rejection are equally important. Silent
fallback would make downstream execution untrustworthy.

## Warnings

- The public `metadata_inspection_only` profile is expected to fail the
  canonical fixture honestly because production consumption capabilities are
  not implemented.
- The test-only declared profile proves deterministic comparison and CLI pass
  wiring, not implementation or execution.
- Contract validation inspects indexed artifact data structurally; Student does
  not consume or expose production training payloads.

## Claims Not Made

Phase 1 does not claim corridor or exemplar payload consumption, loss
computation, model allocation, architecture plugin support, runtime selection,
JAX/XLA portability, checkpoint or schedule execution, training, evaluation,
Hugging Face export, functional-stage distillation, model quality, performance,
scale, or RadLads parity.

## Handoff

Phase 1 is complete. Phase 2 may begin the Student Runtime contract and
orchestration work. Phase 2 must preserve this gate and may not reinterpret a
passing Phase 1 result as proof that training behavior exists.
