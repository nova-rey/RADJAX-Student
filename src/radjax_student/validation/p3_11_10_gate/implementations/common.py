"""Shared, JAX-free provenance primitives for literal P3.11.10B cases."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
)
from radjax_student.validation.p3_11_10_gate.models import ObservedFailure


def callable_identity(value: Callable[..., Any]) -> str:
    """Stable identity including bound public arguments, never a case label."""

    if isinstance(value, functools.partial):
        payload: Mapping[str, Any] = {
            "function": callable_identity(value.func),
            "args": list(value.args),
            "keywords": dict(sorted((value.keywords or {}).items())),
        }
    else:
        payload = {
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        }
    return canonical_digest(payload)


@dataclass(frozen=True)
class GateMutationDelta:
    public_input_kind: str
    canonical_path: str
    operation: str
    before_digest: str
    after_digest: str
    value_summary: Mapping[str, Any]
    baseline_input_digest: str
    mutated_input_digest: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.public_input_kind, self.canonical_path, self.operation)
        ):
            raise ValueError("mutation delta identity fields must be nonempty")
        if self.baseline_input_digest == self.mutated_input_digest:
            raise ValueError("mutation did not change the public input")

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_input_kind": self.public_input_kind,
            "canonical_path": self.canonical_path,
            "operation": self.operation,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "value_summary": dict(self.value_summary),
            "baseline_input_digest": self.baseline_input_digest,
            "mutated_input_digest": self.mutated_input_digest,
        }


@dataclass
class BoundaryProbe:
    """Evidence emitted by the wrapper around the actual public callable."""

    boundary: str
    public_callable: Callable[..., Any]
    public_input_digest: str
    events: list[str] = field(default_factory=list)
    observed_exception: BaseException | None = None
    returned_value: Any = None
    post_boundary_reached: bool = False

    @property
    def callable_identity(self) -> str:
        return callable_identity(self.public_callable)

    @property
    def trace_digest(self) -> str:
        return canonical_digest(
            {
                "boundary": self.boundary,
                "callable": self.callable_identity,
                "input": self.public_input_digest,
                "events": self.events,
                "exception": None
                if self.observed_exception is None
                else type(self.observed_exception).__name__,
                "post_boundary_reached": self.post_boundary_reached,
            }
        )

    def call(self, *args: Any, **kwargs: Any) -> Any:
        self.events.append("invocation_started")
        try:
            self.returned_value = self.public_callable(*args, **kwargs)
        except BaseException as error:
            self.observed_exception = error
            self.events.append("production_exception")
            raise
        self.events.append("invocation_returned")
        return self.returned_value

    def call_catching(self, *args: Any, **kwargs: Any) -> BoundaryProbe:
        """Invoke the selected public callable and retain its real outcome.

        Case implementations do not manufacture trace events or exceptions.  The
        probe is the only component that records invocation, rejection, and the
        post-call sentinel state.
        """

        try:
            self.call(*args, **kwargs)
        except BaseException:
            return self
        self.mark_post_boundary_reached()
        return self

    def mark_post_boundary_reached(self) -> None:
        self.post_boundary_reached = True
        self.events.append("post_boundary_reached")


@dataclass(frozen=True)
class PreparedGateCase:
    baseline_input: Any
    mutated_input: Any
    mutation_delta: GateMutationDelta


@dataclass(frozen=True)
class GateExecutionContext:
    repository_root: Path
    temporary_root: Path


Prepare = Callable[[GateExecutionContext], PreparedGateCase]
Invoke = Callable[[PreparedGateCase, GateExecutionContext], BoundaryProbe]


def prepare_public_input(
    *,
    baseline: Any,
    mutated: Any,
    public_input_kind: str,
    canonical_path: str,
    operation: str,
    value_summary: Mapping[str, Any],
    canonical_baseline: Any | None = None,
    canonical_mutated: Any | None = None,
) -> PreparedGateCase:
    """Bind mutation evidence to the exact object passed to a public API."""

    # Filesystem calls use a normalized public-input representation so random
    # temporary-directory prefixes never leak into recorded evidence.
    baseline_digest = canonical_digest(
        baseline if canonical_baseline is None else canonical_baseline
    )
    mutated_digest = canonical_digest(
        mutated if canonical_mutated is None else canonical_mutated
    )
    delta = GateMutationDelta(
        public_input_kind=public_input_kind,
        canonical_path=canonical_path,
        operation=operation,
        before_digest=baseline_digest,
        after_digest=mutated_digest,
        value_summary=value_summary,
        baseline_input_digest=baseline_digest,
        mutated_input_digest=mutated_digest,
    )
    return PreparedGateCase(baseline, mutated, delta)


@dataclass(frozen=True)
class FailureAdapter:
    exception_type: type[BaseException]
    boundary: str
    code: str
    message_prefix: str | None = None

    def matches(self, error: BaseException, boundary: str) -> bool:
        return (
            isinstance(error, self.exception_type)
            and boundary == self.boundary
            and (
                self.message_prefix is None
                or str(error).startswith(self.message_prefix)
            )
        )


OBSERVED_FAILURE_ADAPTERS: tuple[FailureAdapter, ...] = (
    FailureAdapter(
        ReplayCanonicalError,
        "replay_schema_validation",
        "replay_canonical_error",
    ),
    FailureAdapter(
        ReplayCanonicalError,
        "resume_replay_validation",
        "replay_canonical_error",
    ),
)


def observe_failure(probe: BoundaryProbe) -> ObservedFailure | None:
    """Derive observations solely from a probed public boundary result."""

    error = probe.observed_exception
    if error is None:
        return None
    if hasattr(error, "code"):
        code = str(error.code)  # type: ignore[attr-defined]
    else:
        adapter = next(
            (
                item
                for item in OBSERVED_FAILURE_ADAPTERS
                if item.matches(error, probe.boundary)
            ),
            None,
        )
        if adapter is None:
            code = type(error).__name__
        else:
            code = adapter.code
    return ObservedFailure(
        code=code,
        boundary=probe.boundary,
        exception_type=type(error).__name__,
        phase="public_boundary",
        message_digest=hashlib.sha256(str(error).encode()).hexdigest(),
        details={
            "public_callable": probe.callable_identity,
            "exception_type": type(error).__name__,
        },
    )


def invoke_recorded_positive_control(
    prepared: PreparedGateCase, context: GateExecutionContext
) -> BoundaryProbe:
    """Run a stable accepted public validator for every positive control."""

    from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt

    def parse(_: Any) -> str:
        receipt = StatefulReplayReceipt.from_json_bytes(
            (context.repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
        )
        return str(receipt["evidence_digest"])

    probe = BoundaryProbe(
        "recorded_positive_control", parse, prepared.mutation_delta.mutated_input_digest
    )
    return probe.call_catching(prepared.mutated_input)


@dataclass(frozen=True)
class GateCaseImplementation:
    case_id: str
    section_id: str
    execution_class: str
    expected_boundary: str
    prepare: Prepare
    invoke: Invoke

    @property
    def identity(self) -> str:
        return canonical_digest(
            {
                "case_id": self.case_id,
                "section_id": self.section_id,
                "execution_class": self.execution_class,
                "expected_boundary": self.expected_boundary,
                "prepare": callable_identity(self.prepare),
                "invoke": callable_identity(self.invoke),
            }
        )

    @property
    def effective_identity(self) -> str:
        return canonical_digest(
            {
                "section_id": self.section_id,
                "execution_class": self.execution_class,
                "expected_boundary": self.expected_boundary,
                "prepare": callable_identity(self.prepare),
                "invoke": callable_identity(self.invoke),
            }
        )


__all__ = [
    "BoundaryProbe",
    "FailureAdapter",
    "GateCaseImplementation",
    "GateExecutionContext",
    "GateMutationDelta",
    "OBSERVED_FAILURE_ADAPTERS",
    "PreparedGateCase",
    "callable_identity",
    "observe_failure",
    "invoke_recorded_positive_control",
    "prepare_public_input",
]
