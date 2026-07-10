# P1.8 Student Compatibility Report

P1.8 introduces the first explicit Student-side readiness verdict for a valid
production Tome.

```python
report = evaluate_student_compatibility(view, defaults, capability_profile)
```

The convenience path opens and normalizes the artifact through the existing
P1.6/P1.7 seams:

```python
report = evaluate_tome_path_compatibility(path, capability_profile)
```

Neither API parses producer files directly.

## Capability Profiles

`StudentCapabilityProfile` declares supported contract/Tome/cover versions,
surface kinds and schemas, versioned Contract capabilities, target scopes,
sequence and vocabulary limits, tokenizer identities, producer status policy,
and declarative-plan features.

`metadata_inspection_only_profile()` is the honest default for current Phase 1
behavior. It can inspect the production contract and plan, but declares no
corridor or exemplar payload-consumption capabilities and no sequence/vocab
limits. The canonical fixture therefore fails under this profile with explicit
missing-capability and unevaluated-dimension blockers.

Tests also use a synthetic profile declaring every canonical requirement. That
profile can pass evaluator logic. Its pass proves deterministic comparison, not
payload execution or implementation readiness.

## Report Semantics

`StudentCompatibilityReport` contains:

- pass/fail status and profile ID;
- producer/Contract identity and validation facts;
- required, supported, missing, and execution-unevaluated capabilities;
- supported and unsupported surface IDs;
- sequence and vocabulary checks;
- per-surface target-scope checks;
- declarative pass-plan readiness;
- structured blockers and warnings with stable codes;
- separate artifact and Student claims not made.

Required unknown surfaces, unsupported schemas/scopes, missing capabilities,
unsupported versions, dimension limits, producer/Contract failures, and
unsupported plan metadata are blockers. Unknown optional surfaces remain
warnings. Contract-invalid artifacts fail during P1.6 opening before Student
compatibility evaluation.

## Non-Execution Boundary

Compatibility evaluation does not load payloads, compute losses, allocate a
model, instantiate an architecture, select a runtime, execute checkpoints, or
turn pass records into a schedule. A passing declared profile always carries the
warning and non-claim that capability declaration is not implementation proof.

P1.8 does not claim actual corridor/exemplar consumption, training, checkpoint
execution, JAX portability, runtime readiness, Hugging Face export, model
quality, or parity.
