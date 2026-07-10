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
execution proof remain P2.4 responsibilities.
