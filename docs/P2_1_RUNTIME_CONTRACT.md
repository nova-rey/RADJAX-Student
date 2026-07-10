# P2.1 Runtime Contract and Terminology

P2.1 freezes the architecture-independent runtime boundary before device
inspection or JAX execution is added. It defines the socket; later Phase 2
checkpoints wire the engine.

## Ownership

Runtime answers:

```text
Where does computation happen?
How is it initialized and placed?
How is it compiled and synchronized?
How is execution reported?
```

Runtime owns environment/device observation, backend selection and
initialization, capability declarations, placement and compilation translation,
synchronization, timing, runtime-owned state, diagnostics, reports, and
runtime-specific failures.

Runtime does not own Tome parsing or compatibility, architecture/model
configuration or math, parameter trees, losses, optimizers, data ordering,
training schedules, evaluation policy, or Hugging Face export.

## Public Concepts

`RuntimeConfig` is requested intent: backend/platform preference, precision,
placement, compilation, distributed and fallback policies, required runtime
capabilities, one root seed, and debug intent. It contains no architecture name.

`RuntimeEnvironment` is observed process/runtime state. Unknown JAX versions,
platform, topology counts, and distributed state remain `None`; requested
policy is never rewritten as observed fact.

`DeviceDescriptor` and `DeviceInventory` normalize device and topology facts
without exposing raw backend device objects. Metadata is recursively immutable,
finite JSON data.

`RuntimeCapabilityProfile` is a versioned backend declaration with separate
capabilities and non-capabilities. Capabilities serialize in deterministic
order. Declaration is not execution proof.

`CompilationOptions` exposes only enabled/static/donation/debug/synchronization
intent. Raw `jax.jit` options are not part of the stable contract.

`ExecutionContext` contains backend ID, observed environment, normalized device
inventory, capability declaration, root seed, runtime ID, and runtime metadata.
It is runtime state, not model or optimizer state.

`RuntimeState` reserves a serializable envelope for runtime ID, step, root seed,
config, topology, precision, placement, and resume metadata. P2.1 does not save
or restore it; P2.5 freezes RNG streams and P2.8 implements persistence.

`RuntimeReport` separates status, selected backend/policy, observed facts,
capabilities, blockers, warnings, and claims not made. It serializes without raw
backend objects.

## Policy Vocabulary

Precision:

```text
float32 bfloat16 float16 mixed automatic unspecified
```

Placement:

```text
single_device replicated data_sharded model_sharded automatic unspecified
```

Compilation:

```text
eager jit automatic unspecified
```

Distributed request:

```text
disabled auto required
```

Fallback:

```text
disallowed allow_compatible
```

Fallback is disallowed by default. A backend must never silently replace a
requested accelerator with CPU. A future selector may use a compatible fallback
only when policy permits it and the report makes that decision explicit.

## Capability Vocabulary

P2.1 reserves these architecture-independent capability strings:

```text
runtime.single_process_v1
runtime.multi_process_v1
placement.single_device_v1
placement.replicated_v1
placement.data_sharded_v1
placement.model_sharded_v1
compilation.jit_v1
execution.synchronize_v1
state.runtime_envelope_v1
```

The version suffix makes semantic change visible. Backends must reject missing
requirements explicitly and declare non-capabilities honestly.

## Backend Protocol

`RuntimeBackend` is a small structural protocol:

```text
backend_id
inspect_environment()
capability_profile()
initialize(config)
place(value, placement)
compile(function, options)
synchronize(value)
close(context)
```

P2.1 includes no concrete backend and no registry. P2.2 implements observation;
P2.3 implements registration/selection; P2.4 adds the first CPU execution smoke.

Selection will consider requested backend/platform, observed environment,
required capabilities, and fallback policy. It must return a selected backend
plus report or explicit structured blockers; global JAX defaults are not a
selection API.

## Structured Failures

Runtime errors carry a stable code, message, and immutable JSON details:

```text
runtime_backend_not_found
runtime_backend_unavailable
requested_platform_unavailable
runtime_capability_missing
runtime_initialization_failed
runtime_configuration_invalid
runtime_environment_incompatible
runtime_fallback_disallowed
runtime_internal_error
```

`RuntimeContractError` and `RuntimeIssue` preserve this structure through
exceptions and reports. Vague unstructured fallback is outside the contract.

## Import Boundary

Generic runtime modules use only the Python standard library. They do not import
JAX/JAXLIB, NumPy, Contract/Tome, artifacts, architecture/students, losses,
training, schedules, Hugging Face integrations, or optional ML stacks. A future
JAX backend may import JAX behind its implementation boundary without changing
the generic models.

Importing `radjax_student.runtime` does not inspect devices or initialize JAX.

## Validation

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_runtime_contract.py
```

The tests lock serialization, unknown values, immutability, vocabulary,
structured errors, protocol conformance, generic import boundaries, and the
absence of model/training execution. The complete Phase 1 acceptance gate also
remains required.

## Claim

P2.1 claims only:

```text
RADJAX-Student has a stable, architecture-independent contract
for runtime configuration, observation, capabilities, execution context,
errors, and reporting.
```

P2.1 does not claim JAX installation or inspection, backend availability or
selection, placement, compilation, synchronization, GPU/TPU/distributed
execution, finalized RNG streams, state persistence, architecture support,
payload loading, training, evaluation, export, performance, or model quality.
