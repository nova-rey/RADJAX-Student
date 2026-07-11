# P3.8 Metrics, Hooks, And Reporting

P3.8 keeps observability subordinate to generic learning execution.

- P3.8A defines deterministic observer-only hook dispatch and failure policy.
- P3.8B places those hooks at generic loop lifecycle boundaries.
- P3.8C describes a completed loop with a deterministic immutable report.

Metrics are retained as bounded loop history. Hooks can observe lifecycle
contexts and return metrics, warnings, or blockers under their established
policy, but cannot mutate learning state or direct the loop. Reporting converts
the completed `LearningLoopResult` after execution and cannot become a loop
control surface.

P3.8C reports lifecycle order, retained metric summaries, warning and blocker
occurrences, checkpoint receipt order, and public scope kinds. It provides no
external telemetry, evaluation, dashboards, architecture-specific fields, or
Tome-specific reporting.

P3.8D closes P3.8 with one deterministic acceptance receipt. The gate audits
the existing metric, hook, loop, and report behavior without creating a second
control plane. P3.9 synthetic learning smoke is now unblocked.
