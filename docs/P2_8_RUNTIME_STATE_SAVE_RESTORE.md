# P2.8 Runtime State Save/Restore Foundation

P2.8 defines the first persistent runtime-owned artifact. The artifact is small,
portable, backend-neutral, versioned, human-inspectable, and deterministic.

## Envelope

`RuntimeState` serializes `runtime_state.v1` with runtime ID, global step, root
seed, complete immutable `RuntimeKeys` lineage, `RuntimeConfig`, environment and
topology summaries, precision and placement policy, backend ID, generic resume
metadata, and explicit non-claims.

It does not serialize model parameters, optimizer state, training batches, Tome
payloads, architecture state, compiled executables, raw devices, or raw JAX key
arrays. Future checkpoint contracts must add their own explicit aggregate rather
than extending this runtime envelope implicitly.

## Format and Integrity

The file layout is deliberately boring:

```text
runtime_state/
  runtime_state.json
  manifest.json
```

JSON is UTF-8, sorted-key, compact, newline-terminated, finite, and stable.
`manifest.json` is written last and records the state file size plus SHA-256
digest. It also hashes its own base manifest payload. Restore validates regular
files, internal paths, manifest schema, file names, sizes, digests, top-level
state fields, and the deterministic RNG tree before returning a typed state.

`save_runtime_state()` refuses an existing output directory unless
`overwrite=True`; explicit overwrite refuses directories containing unrelated or
unsafe entries. `load_runtime_state()` returns state only after validation;
`load_runtime_state_with_receipt()` additionally returns verified-file metadata.

## Resume Compatibility

`evaluate_runtime_resume_compatibility()` compares runtime facts only: backend
ID, precision, placement, distributed policy, and root seed. Observed topology
differences are warnings because saved topology is historical metadata, not proof
that the current machine is equivalent. It never compares architecture shape or
model state.

## Doctor Smoke

`radjax-student doctor --runtime-state-smoke` is opt-in. It uses a temporary
directory and proves this narrow chain:

```text
CPU runtime smoke
-> create runtime metadata at step 3
-> save
-> load and validate
-> compare runtime intent
-> continuation CPU runtime smoke
```

The continuation proves only metadata continuity. It does not restore a model,
optimizer, devices, executables, training cursor, topology migration, or full
run reproducibility.

## Claims

P2.8 claims RADJAX-Student can save, validate, restore, and compare a versioned
runtime-owned state envelope containing runtime identity, policy, step, RNG
lineage, and topology metadata without storing model or optimizer state.
