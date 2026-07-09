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
