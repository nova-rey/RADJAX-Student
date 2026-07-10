# Validation

Validation is the boundary between artifact trust and runtime action.

This package is the long-term home for Student-side compatibility and readiness
checks. Contract owns shared schemas and artifact validation; Student validation
should state whether this runtime can consume a valid Contract artifact.

`run_defaults` is intentionally pre-compatibility. It reports artifact facts,
surface and capability requirements, declarative pass intent, user choices, and
later-phase policy gaps.

`compatibility` owns the first readiness decision. It evaluates those facts
against an explicit declaration profile and emits stable blockers and warnings.
It does not prove execution support or instantiate runtime/architecture code.
