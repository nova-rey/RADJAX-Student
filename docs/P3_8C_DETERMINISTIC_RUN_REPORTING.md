# P3.8C Deterministic Run Reporting

P3.8C adds a pure report of a completed generic learning loop. It does not
participate in execution, change loop control flow, or stream information to an
external service.

## Report Contract

`build_learning_run_report()` converts an existing `LearningLoopResult` into an
immutable `LearningRunReport` with schema version
`radjax.learning_run_report.v1`. The report contains:

- `run_id` and validated pass/fail status, stop reason, completed steps, and
  global step;
- name-sorted metric summaries;
- ordered lifecycle events and their first/last event;
- separate warning-code and hook-blocker-code occurrence lists;
- ordered checkpoint receipts;
- scalar update and objective scope kinds;
- original structured hook blockers, fixed claims and non-claims, and immutable
  JSON-safe metadata.

The conversion is pure. It does not write files, make network calls, mutate the
loop result, or serialize parameters, gradients, optimizer state, architecture
state, runtime handles, raw batches, or traceback data.

## Metrics And Ordering

Each metric name is summarized with `count`, `last`, `minimum`, `maximum`,
`mean`, and `sum`. Names are ordered lexicographically, while observations for a
given name retain their loop-result order so `last` remains meaningful.

`LearningLoopResult.metrics` is retained history, not a complete-run metric
ledger. Every report therefore declares
`metric_summary_source: "bounded_history"`. The report does not claim complete
run statistics when the loop has discarded older observations.

Lifecycle events, warning occurrences, hook blocker occurrences, and checkpoint
receipts preserve their original order. Warnings and blockers remain distinct:
a warning code is not represented as a blocking outcome.

## Deterministic JSON

`to_dict()` produces only immutable report-owned values. `to_json()` uses sorted
keys and compact separators, making repeated serialization of the same report
byte-stable.

The report claims only generic report generation, deterministic serialization,
bounded metric reporting, and observer-only hook reporting. It explicitly does
not claim model quality, real architecture support, Tome training, language
modeling, distributed training, accelerator performance, external telemetry, or
evaluation.

## Optional Loop Integration

`run_learning_loop(..., emit_run_report=False)` keeps the existing default path:
the returned `LearningLoopResult.report` is `None`. With
`emit_run_report=True`, the loop first completes normally and only then derives
the report from its completed result. Report generation cannot alter the
completed learning result; an invalid report conversion leaves the completed
result intact with no attached report.

No external telemetry, dashboard, streaming, evaluation, or reporting control
surface is included.
