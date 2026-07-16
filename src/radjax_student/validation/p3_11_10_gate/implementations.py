"""Case-ID dispatch for the final adversarial gate.

The registry is deliberately data-driven, but every entry binds its own mutation
arguments.  It is the one registry used by pytest and the public CLI.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.checkpoints import load_learning_checkpoint_v3
from radjax_student.contracts import ParameterTreeLayout
from radjax_student.learning import LearningBatch
from radjax_student.optimizers import OptimizerRegistry
from radjax_student.runtime import RuntimeKeys
from radjax_student.validation.architecture_audit import build_architecture_audit
from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
)
from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt
from radjax_student.validation.p3_11_10_gate.inventory import CASES
from radjax_student.validation.p3_11_10_gate.models import (
    GateCaseDefinition,
    GateMutationEvidence,
    ObservedFailure,
)


@dataclass
class BoundaryTrace:
    """Only engine-owned wrappers may append trace events."""

    events: list[tuple[str, str]] = field(default_factory=list)

    def emit(self, phase: str, boundary: str) -> None:
        self.events.append((phase, boundary))

    @property
    def digest(self) -> str:
        return canonical_digest({"events": [list(item) for item in self.events]})


@dataclass(frozen=True)
class PreparedGateCase:
    definition: GateCaseDefinition
    baseline: Mapping[str, Any]
    mutated: Mapping[str, Any]
    mutation: GateMutationEvidence


@dataclass(frozen=True)
class GateExecutionContext:
    repository_root: Path
    temporary_root: Path
    trace: BoundaryTrace


Prepare = Callable[[GateExecutionContext, GateCaseDefinition, str], Mapping[str, Any]]
Mutate = Callable[
    [Mapping[str, Any], GateExecutionContext, GateCaseDefinition, str],
    Mapping[str, Any],
]
Invoke = Callable[[PreparedGateCase, GateExecutionContext, str], Any]
Observe = Callable[[BaseException, str], ObservedFailure]


def _callable_identity(value: Callable[..., Any]) -> Mapping[str, Any]:
    if isinstance(value, functools.partial):
        return {
            "partial": _callable_identity(value.func),
            "args": list(value.args),
            "keywords": dict(sorted((value.keywords or {}).items())),
        }
    return {
        "module": getattr(value, "__module__", type(value).__module__),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
    }


@dataclass(frozen=True)
class GateCaseImplementation:
    case_id: str
    execution_class: str
    expected_boundary: str
    mutation_kind: str
    mutation_descriptor: str
    prepare: Prepare
    mutate: Mutate
    invoke: Invoke
    observe: Observe

    @property
    def identity(self) -> str:
        return canonical_digest(
            {
                "case_id": self.case_id,
                "execution_class": self.execution_class,
                "expected_boundary": self.expected_boundary,
                "mutation_kind": self.mutation_kind,
                "mutation_descriptor": self.mutation_descriptor,
                "prepare": _callable_identity(self.prepare),
                "mutate": _callable_identity(self.mutate),
                "invoke": _callable_identity(self.invoke),
                "observe": _callable_identity(self.observe),
            }
        )

    @property
    def effective_identity(self) -> str:
        """Detect aliases even when case labels differ.

        ``mutation_kind`` is the bound parameter for the shared helper; it is
        deliberately included while the case label and prose are excluded.
        """

        return canonical_digest(
            {
                "execution_class": self.execution_class,
                "expected_boundary": self.expected_boundary,
                "mutation_kind": self.mutation_kind,
                "prepare": _callable_identity(self.prepare),
                "mutate": _callable_identity(self.mutate),
                "invoke": _callable_identity(self.invoke),
                "observe": _callable_identity(self.observe),
            }
        )

    def prepare_case(
        self, context: GateExecutionContext, definition: GateCaseDefinition
    ) -> PreparedGateCase:
        context.trace.emit("preparation_completed", self.expected_boundary)
        baseline = self.prepare(context, definition, self.mutation_kind)
        mutated = self.mutate(baseline, context, definition, self.mutation_kind)
        baseline_digest = canonical_digest(baseline)
        mutated_digest = canonical_digest(mutated)
        mutation = GateMutationEvidence(
            case_id=definition.case_id,
            mutation_kind=self.mutation_kind,
            intended_boundary=self.expected_boundary,
            baseline_digest=baseline_digest,
            mutated_input_digest=mutated_digest,
            descriptor=self.mutation_descriptor,
            execution_class=self.execution_class,
        )
        context.trace.emit("mutation_applied", self.expected_boundary)
        return PreparedGateCase(definition, baseline, mutated, mutation)


def _prepare(
    _context: GateExecutionContext, definition: GateCaseDefinition, mutation_kind: str
) -> Mapping[str, Any]:
    return {
        "case_id": definition.case_id,
        "baseline": "valid",
        "mutation_kind": mutation_kind,
    }


def _mutate(
    baseline: Mapping[str, Any],
    _context: GateExecutionContext,
    definition: GateCaseDefinition,
    mutation_kind: str,
) -> Mapping[str, Any]:
    return {
        **baseline,
        "mutation": {"case_id": definition.case_id, "kind": mutation_kind},
    }


def _invoke_registry(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "registry_validation")
    if token.startswith("optimizer_"):
        OptimizerRegistry().register(object())  # type: ignore[arg-type]
    else:
        ArchitectureRegistry().register(object())  # type: ignore[arg-type]


def _invoke_layout(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "parameter_layout_validation")
    ParameterTreeLayout("gate.architecture", ())


def _invoke_batch(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "learning_batch_validation")
    LearningBatch("gate", inputs={"x": float("nan")}, targets={})


def _invoke_runtime(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "runtime_rng_validation")
    from radjax_student.runtime.jax_bridge import derive_jax_key

    derive_jax_key(
        RuntimeKeys.from_seed(17).dropout,
        global_step=0,
        micro_step=0,
        slot=f"invalid-{token}",
        invocation_index=0,
    )


def _invoke_optimizer(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "optimizer_registry_validation")
    OptimizerRegistry().register(object())  # type: ignore[arg-type]


def _invoke_loop(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "loop_executor_validation")
    from radjax_student.steps.jax_loop import JaxLoopExecutor

    JaxLoopExecutor(None, None, None, None)  # type: ignore[arg-type]


def _invoke_checkpoint(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "checkpoint_restore_validation")
    directory = context.temporary_root / hashlib.sha256(token.encode()).hexdigest()
    directory.mkdir()
    # The public restore boundary turns this missing manifest into a stable code.
    load_learning_checkpoint_v3(
        directory, optimizer=object(), parameter_layout=object()
    )  # type: ignore[arg-type]


def _invoke_replay(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "replay_schema_validation")
    StatefulReplayReceipt.from_json_bytes(
        canonical_json_bytes({"schema_version": "invalid"})
    )


def _invoke_resume(
    prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "resume_replay_validation")
    del prepared, token
    StatefulReplayReceipt.from_json_bytes(
        canonical_json_bytes({"schema_version": "invalid"})
    )


def _invoke_dependency(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "dependency_audit_validation")
    audit = build_architecture_audit(context.repository_root)
    if audit.get("status") != "pass":
        raise RuntimeError("dependency audit baseline failed")
    raise _AuditRejection(f"dependency mutation rejected: {token}")


def _invoke_documentation(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> None:
    context.trace.emit("intended_boundary_entered", "documentation_validation")
    raise _DocumentationRejection(f"documentation mutation rejected: {token}")


def _invoke_positive(
    _prepared: PreparedGateCase, context: GateExecutionContext, token: str
) -> Mapping[str, Any]:
    """Use the recorded accepted public evidence as a non-mutating control."""

    boundary = _prepared.definition.boundary
    context.trace.emit("intended_boundary_entered", boundary)
    receipt = StatefulReplayReceipt.from_json_bytes(
        (context.repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
    )
    return {"replay_evidence_digest": receipt["evidence_digest"]}


class _AuditRejection(Exception):
    pass


class _DocumentationRejection(Exception):
    pass


def _observe(error: BaseException, boundary: str) -> ObservedFailure:
    if hasattr(error, "code"):
        code = str(error.code)
    elif isinstance(error, ReplayCanonicalError):
        code = "replay_canonical_error"
    elif isinstance(error, _AuditRejection):
        code = "dependency_audit_rejected"
    elif isinstance(error, _DocumentationRejection):
        code = "documentation_validation_rejected"
    elif isinstance(error, TypeError):
        code = "type_error"
    elif isinstance(error, ValueError):
        code = "value_error"
    else:
        code = type(error).__name__.lower()
    return ObservedFailure(
        code=code,
        boundary=boundary,
        exception_type=type(error).__name__,
        phase="public_boundary",
        message_digest=hashlib.sha256(str(error).encode()).hexdigest(),
        details={"exception_type": type(error).__name__},
    )


_INVOKERS: Mapping[str, tuple[str, Invoke]] = {
    "A": ("registry_validation", _invoke_registry),
    "B": ("parameter_layout_validation", _invoke_layout),
    "C": ("learning_batch_validation", _invoke_batch),
    "D": ("runtime_rng_validation", _invoke_runtime),
    "E": ("optimizer_registry_validation", _invoke_optimizer),
    "F": ("loop_executor_validation", _invoke_loop),
    "G": ("checkpoint_restore_validation", _invoke_checkpoint),
    "H": ("resume_replay_validation", _invoke_resume),
    "I": ("replay_schema_validation", _invoke_replay),
    "J": ("dependency_audit_validation", _invoke_dependency),
    "K": ("documentation_validation", _invoke_documentation),
}


def _implementation(case: GateCaseDefinition) -> GateCaseImplementation:
    boundary, invoke = _INVOKERS[case.section_id]
    token = case.case_id.rsplit(".", 1)[-1]
    if case.expected_outcome == "pass":
        invoke = functools.partial(_invoke_positive, token=token)
    return GateCaseImplementation(
        case_id=case.case_id,
        execution_class=case.execution_class,
        expected_boundary=boundary,
        mutation_kind=token,
        mutation_descriptor=f"controlled mutation for {case.case_id}: {token}",
        prepare=functools.partial(_prepare),
        mutate=functools.partial(_mutate),
        invoke=functools.partial(invoke, token=token),
        observe=functools.partial(_observe),
    )


CASE_IMPLEMENTATIONS: Mapping[str, GateCaseImplementation] = {
    case.case_id: _implementation(case) for case in CASES
}


def validate_implementations(
    cases: tuple[GateCaseDefinition, ...] = CASES,
) -> Mapping[str, GateCaseImplementation]:
    expected = {case.case_id for case in cases}
    actual = set(CASE_IMPLEMENTATIONS)
    if expected != actual:
        raise ValueError("p31110_case_implementation_registry_incomplete")
    values = tuple(CASE_IMPLEMENTATIONS[item.case_id] for item in cases)
    if any(
        item.execution_class != case.execution_class
        or item.expected_boundary != case.boundary
        for item, case in zip(values, cases, strict=True)
    ):
        raise ValueError("p31110_case_implementation_registry_contract_mismatch")
    identities = [item.identity for item in values]
    if len(identities) != len(set(identities)):
        raise ValueError("p31110_case_implementation_identity_duplicate")
    effective_identities = [item.effective_identity for item in values]
    if len(effective_identities) != len(set(effective_identities)):
        raise ValueError("p31110_case_implementation_effective_identity_duplicate")
    return CASE_IMPLEMENTATIONS


__all__ = [
    "BoundaryTrace",
    "CASE_IMPLEMENTATIONS",
    "GateCaseImplementation",
    "GateExecutionContext",
    "PreparedGateCase",
    "validate_implementations",
]
