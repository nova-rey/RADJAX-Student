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
