# P3.8D Observability Golden Acceptance Gate

P3.8D closes metrics, hooks, loop integration, and run reporting with one
deterministic in-memory audit. It adds no new learning or telemetry capability.

## Receipt

`P38ObservabilityAcceptanceReceipt` uses schema
`radjax.p3_8_observability_acceptance.v1`. A passing receipt requires every
validity flag to be true and no blockers. A failing receipt retains stable,
structured blockers for the failed audit section.

The gate validates metrics, hook dispatch, lifecycle integration, run reports,
deterministic replay, failure paths, observer-only data boundaries, bounded
history honesty, import boundaries, documentation, and test inventory. Its
stable blocker codes begin with `p3_8_` and identify the failed section without
serializing a traceback or mutable learning state.

## Guarantees

Replay compares status, stop reason, invocation-local and global steps, bounded
metrics, checkpoint receipts, warnings, lifecycle events, hook blockers, and
the deterministic report dictionary and JSON. The gate explicitly checks that
reports use bounded retained observations, not an invented complete history.

Hooks and reports remain observer-only: no parameters, gradients, optimizer or
architecture state, runtime handles, raw batches, or mutable scope control is
exposed. The audit also rejects optional ML, telemetry, network, and Tome
producer imports from observability sources.

## Run

```bash
PYTHONPATH=src python3 -m radjax_student.learning.observability_acceptance
PYTHONPATH=src python3 -m radjax_student.learning.observability_acceptance --json
```

The command is offline, writes no files, exits zero only for a passing receipt,
and prints no model or report payloads.

## Adversarial Hardening

Each audit uses immutable injectable production seams. Negative tests introduce
broken retention, summaries, hook dispatches, loop results, report builders,
source loaders, and path checks, then prove the corresponding validity flag and
stable blocker fail.

Import and test-inventory checks use Python AST inspection, covering aliased
imports, dynamic import calls, test-function imports/calls, `assert True`,
`pytest.skip`, and bare `pass` statements in test bodies. Unexpected audit
exceptions fail closed: the affected section is false and the receipt includes
`p3_8_internal_error` with only section and exception type details.

## Final Coverage Closure

The gate also audits explicit hook warnings, failure codes, and disablement
actions. Run-report coverage verifies checkpoint receipt order, public update
and objective scopes, construction only from a completed result, and isolation
when report construction fails. Observer-only coverage compares the complete
finished-run state and rejects report JSON containing forbidden state names.

The report-construction seam temporarily substitutes only the builder imported
inside the public opt-in loop, restores it in `finally`, and exercises
`run_learning_loop(..., emit_run_report=True)` directly. This proves both the
single post-completion builder call and the public catch-and-return path when
the builder fails.

Import and test-placeholder regressions are injected through the complete
receipt, not accepted merely because a helper parser detects them. Each such
regression must make its receipt section false and emit the corresponding stable
blocker.

## Non-Claims

P3.8D does not claim model quality, a real architecture, Tome training,
language modeling, distributed training, accelerator performance, external
telemetry, or evaluation. P3.9 is unblocked only after this receipt passes.
