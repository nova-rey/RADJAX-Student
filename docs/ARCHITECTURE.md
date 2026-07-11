# Architecture

RADJAX-Student is the student-side artifact consumer, training, evaluation, and
export package.

It consumes artifacts defined by RADJAX-Contract and produced by RADJAX-Tome. It
must not import Tome directly.

The normative architecture boundary document is
[ARCHITECTURE_CHARTER.md](ARCHITECTURE_CHARTER.md). Future implementation should
use that charter to decide whether a feature belongs in artifacts, runtime,
architecture plugins, training, schedules, Hugging Face export, reports, CLI, or
validation.

The practical product boundary is documented in
[STUDENT_SPLIT_CONTRACT.md](STUDENT_SPLIT_CONTRACT.md). That contract states
what RADJAX-Student owns, what it does not own, and what the project may and may
not claim during foundation work.

## Production Artifact Boundary

`open_tome_artifact()` is the stable Student-owned entry point. Its production
path calls RADJAX-Contract parsing, validation, and inspection APIs and returns
immutable normalized metadata. Student does not parse producer JSON, walk Tome
directories, guess filenames, or require a production `manifest.json`.

The root view exposes an arbitrary surface collection and the Contract-validated
pass plan. Corridor and exemplar accessors are optional metadata projections;
they do not load assignment arrays, selected payload records, models, runtimes,
or schedules. The older dense-logits manifest path remains explicitly
`legacy_dense_v0` smoke/debug support.

`infer_run_defaults()` is the next metadata-only station. It transforms the
validated view into artifact facts, arbitrary surface summaries, capability
requirements, and declarative passes. It does not decide compatibility or turn
pass metadata into a schedule. User input remains limited to architecture,
architecture size/config, training budget, and output location; runtime and
training policy remain owned by later phases.

`evaluate_student_compatibility()` is the first Student readiness gate. Contract
still owns structural artifact validity; Student compares normalized requirements
with an immutable capability profile. The report can pass or fail declared
readiness, but it does not instantiate an architecture or prove that declared
capabilities execute correctly.

## CLI and Reports

`radjax-student inspect` is a thin adapter over the same public pipeline:

```text
open_tome_artifact()
-> infer_run_defaults()
-> evaluate_student_compatibility()
-> StudentInspectionReport
-> human or JSON renderer
```

Artifact parsing and validation remain in Contract-backed artifact APIs;
default inference and compatibility policy remain in `validation/`; stable
aggregates live in `reports/`; argument parsing, rendering, and exit-code mapping
live in `cli/`. `radjax-student doctor` composes those same boundaries against
the accepted fixture. Neither command owns producer schemas, loads Student
training payloads, allocates a model, or executes a runtime or schedule.

## Runtime Contract

P2.1 defines runtime as an architecture-independent boundary under `runtime/`.
Requested `RuntimeConfig` policy remains distinct from observed
`RuntimeEnvironment` and `DeviceInventory` facts. Versioned capability profiles,
structured errors, execution contexts, state envelopes, and reports serialize
without raw backend objects.

The `RuntimeBackend` protocol describes environment inspection, capability
declaration, initialization, placement, compilation, synchronization, and
teardown. P2.1 provides no concrete backend or registry and performs none of
those actions. Generic runtime modules use only the Python standard library and
do not depend on artifacts, architecture, training, schedules, JAX, or optional
ML packages.

P2.2 implements only the protocol's observation side as the standalone
`inspect_runtime_environment()` API. JAX/JAXLIB imports are lazy, visible
devices normalize into serializable descriptors, and unknown facts stay
unknown.

P2.3 consumes those supplied facts through an instance-owned backend registry
and a pure selection function. Backend declarations remain serializable and
selection never initializes a context, imports JAX, allocates an array, places a
value, compiles, synchronizes, or executes. `ExecutionContext` and the first
execution proof were deferred to P2.4.

P2.4 adds only the selected JAX CPU heartbeat in `runtime/smoke.py`. It consumes
the existing inspection and selection result, builds a runtime-owned context,
places one small host value on the selected CPU explicitly, executes one eager
pure function, synchronizes, validates its host result, and closes in `finally`.
The module has no architecture, artifact, loss, optimizer, schedule, or training
dependency; broader execution policies remain later checkpoints.

P2.5 keeps deterministic randomness in the runtime boundary through a versioned
`RuntimeKeys` tree attached to every `ExecutionContext`. Its fixed named streams
are portable lineage metadata, not backend RNG objects. Architecture and training
code must later consume supplied streams rather than creating independent RNG
state.

P2.6 adds a topology-free placement language at the same runtime boundary.
Architecture plugins may declare semantic logical axes and stable value paths;
runtime retains ownership of future device/mesh translation. Plans are immutable
intent with no JAX sharding object, device list, model tree, or execution path.

P2.7 gives runtime the only public pure-function execution surface. Callers
provide stable function/request intent; backend adapters own compilation,
dispatch, synchronization, and opaque handles. This keeps raw JAX JIT options,
timing policy, and argument donation out of architecture and training code.

P2.8 gives runtime the only persistence surface for runtime identity and policy.
The envelope records metadata, not model or optimizer trees: future checkpoint
contracts may extend it through explicit ownership boundaries, but they cannot
turn this runtime artifact into architecture-specific state. Restore validates
continuity and compatibility; it does not prove equivalent execution.

P2.9 keeps accelerator behavior in the same runtime path. Architecture plugins
do not branch by CPU/GPU/TPU: runtime selects one observed device, places one
small value, executes the shared pure function, validates state continuity, and
emits a target receipt. Absent hardware remains a runtime observation, not an
architecture failure.

P2.10 closes the runtime foundation with an acceptance gate rather than a new
runtime abstraction. Its receipt records the validated seams and non-claims, so
later generic training work can depend on the runtime boundary without treating
architecture, model state, or performance as already proven.
