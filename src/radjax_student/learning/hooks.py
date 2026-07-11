"""Deterministic observer-only lifecycle hooks; loop integration is deferred."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from radjax_student.learning._json import (
    freeze_json_mapping,
    json_value,
    unique_strings,
)
from radjax_student.learning.errors import LearningIssue
from radjax_student.learning.models import MetricRecord

HOOK_EVENTS = (
    "loop_start",
    "batch_received",
    "step_start",
    "step_end",
    "checkpoint",
    "loop_end",
    "failure",
)
_CLAIMS = ("hooks_do_not_mutate_core_state", "loop_integration_deferred")


@dataclass(frozen=True)
class HookContext:
    run_id: str
    event_type: str
    event_sequence: int
    global_step: int
    metrics: tuple[MetricRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    claims_not_made: tuple[str, ...] = _CLAIMS

    def __post_init__(self):
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or self.event_type not in HOOK_EVENTS
            or type(self.event_sequence) is not int
            or self.event_sequence < 0
            or type(self.global_step) is not int
            or self.global_step < 0
        ):
            raise ValueError("learning_hook_context_invalid")
        if any(not isinstance(x, MetricRecord) for x in self.metrics):
            raise TypeError("metrics must contain MetricRecord")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "event_type": self.event_type,
            "event_sequence": self.event_sequence,
            "global_step": self.global_step,
            "metrics": [x.to_dict() for x in self.metrics],
            "metadata": json_value(self.metadata),
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class HookResult:
    status: Literal["pass", "warning", "fail"] = "pass"
    metrics: tuple[MetricRecord, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    claims_not_made: tuple[str, ...] = _CLAIMS

    def __post_init__(self):
        if self.status not in ("pass", "warning", "fail") or (
            self.status == "warning" and not self.warnings
        ):
            raise ValueError("learning_hook_result_invalid")
        if any(not isinstance(x, MetricRecord) for x in self.metrics) or any(
            not isinstance(x, LearningIssue) for x in self.warnings
        ):
            raise TypeError("invalid hook result contents")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self):
        return {
            "status": self.status,
            "metrics": [x.to_dict() for x in self.metrics],
            "warnings": [x.to_dict() for x in self.warnings],
            "metadata": json_value(self.metadata),
            "claims_not_made": list(self.claims_not_made),
        }


class LearningHook(Protocol):
    hook_id: str
    priority: int
    supported_events: tuple[str, ...]

    def on_event(self, context: HookContext) -> HookResult: ...


@dataclass(frozen=True)
class HookRegistration:
    hook_id: str
    priority: int
    enabled: bool = True
    supported_events: tuple[str, ...] = HOOK_EVENTS
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if (
            not isinstance(self.hook_id, str)
            or not self.hook_id
            or type(self.priority) is not int
            or not isinstance(self.enabled, bool)
            or any(x not in HOOK_EVENTS for x in self.supported_events)
        ):
            raise ValueError("learning_hook_invalid")
        object.__setattr__(
            self,
            "supported_events",
            unique_strings(self.supported_events, "supported_events"),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self):
        return {
            "hook_id": self.hook_id,
            "priority": self.priority,
            "enabled": self.enabled,
            "supported_events": list(self.supported_events),
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class HookPolicy:
    failure_mode: Literal["fail_fast", "warn_and_continue", "disable_hook"] = (
        "fail_fast"
    )
    allow_metric_emission: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if self.failure_mode not in (
            "fail_fast",
            "warn_and_continue",
            "disable_hook",
        ) or not isinstance(self.allow_metric_emission, bool):
            raise ValueError("learning_hook_invalid")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self):
        return {
            "failure_mode": self.failure_mode,
            "allow_metric_emission": self.allow_metric_emission,
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class HookExecutionReceipt:
    hook_id: str
    event_type: str
    event_sequence: int
    status: str
    metrics_emitted: int = 0
    warning_codes: tuple[str, ...] = ()
    disabled_after_failure: bool = False
    failure_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if (
            not isinstance(self.hook_id, str)
            or not self.hook_id
            or self.event_type not in HOOK_EVENTS
            or type(self.event_sequence) is not int
            or self.event_sequence < 0
            or self.status not in ("pass", "warning", "fail")
            or type(self.metrics_emitted) is not int
            or self.metrics_emitted < 0
            or not isinstance(self.disabled_after_failure, bool)
            or (
                self.failure_code is not None
                and (not isinstance(self.failure_code, str) or not self.failure_code)
            )
        ):
            raise ValueError("learning_hook_invalid")
        object.__setattr__(
            self, "warning_codes", unique_strings(self.warning_codes, "warning_codes")
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self):
        return {
            "hook_id": self.hook_id,
            "event_type": self.event_type,
            "event_sequence": self.event_sequence,
            "status": self.status,
            "metrics_emitted": self.metrics_emitted,
            "warning_codes": list(self.warning_codes),
            "disabled_after_failure": self.disabled_after_failure,
            "failure_code": self.failure_code,
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class HookDispatchResult:
    status: Literal["pass", "warning", "fail"]
    receipts: tuple[HookExecutionReceipt, ...]
    metrics: tuple[MetricRecord, ...]
    warnings: tuple[LearningIssue, ...]
    blockers: tuple[LearningIssue, ...]
    disabled_hook_ids: tuple[str, ...]

    def __post_init__(self):
        if (
            self.status not in ("pass", "warning", "fail")
            or any(not isinstance(x, HookExecutionReceipt) for x in self.receipts)
            or any(not isinstance(x, MetricRecord) for x in self.metrics)
            or any(
                not isinstance(x, LearningIssue)
                for x in (*self.warnings, *self.blockers)
            )
        ):
            raise ValueError("learning_hook_dispatch_failed")
        if (
            (self.status == "fail" and not self.blockers)
            or (self.status == "pass" and (self.blockers or self.warnings))
            or (self.status == "warning" and (not self.warnings or self.blockers))
        ):
            raise ValueError("learning_hook_dispatch_failed")
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "blockers", tuple(self.blockers))
        object.__setattr__(
            self,
            "disabled_hook_ids",
            tuple(sorted(unique_strings(self.disabled_hook_ids, "disabled_hook_ids"))),
        )

    def to_dict(self):
        return {
            "status": self.status,
            "receipts": [x.to_dict() for x in self.receipts],
            "metrics": [x.to_dict() for x in self.metrics],
            "warnings": [x.to_dict() for x in self.warnings],
            "blockers": [x.to_dict() for x in self.blockers],
            "disabled_hook_ids": list(self.disabled_hook_ids),
        }


def dispatch_hooks(
    hooks, policy: HookPolicy, context: HookContext, disabled_hook_ids=()
):
    for hook in hooks:
        if (
            not isinstance(getattr(hook, "hook_id", None), str)
            or not hook.hook_id
            or type(getattr(hook, "priority", None)) is not int
            or not isinstance(getattr(hook, "supported_events", None), (tuple, list))
            or len(set(hook.supported_events)) != len(hook.supported_events)
            or any(event not in HOOK_EVENTS for event in hook.supported_events)
            or not callable(getattr(hook, "on_event", None))
        ):
            raise ValueError("learning_hook_invalid")
    disabled = set(disabled_hook_ids)
    ordered = sorted(hooks, key=lambda h: (h.priority, h.hook_id))
    if len({h.hook_id for h in ordered}) != len(ordered):
        raise ValueError("learning_hook_duplicate")
    receipts = []
    metrics = []
    warnings = []
    blockers = []
    for hook in ordered:
        if hook.hook_id in disabled or context.event_type not in hook.supported_events:
            continue
        try:
            result = hook.on_event(context)
        except Exception as exc:
            result = None
            code = "learning_hook_failed"
            details = {"hook_id": hook.hook_id, "exception_type": type(exc).__name__}
        else:
            if not isinstance(result, HookResult):
                code = "learning_hook_result_invalid"
                details = {
                    "hook_id": hook.hook_id,
                    "returned_type": type(result).__name__,
                }
                result = None
            elif result.status == "fail":
                code = "learning_hook_failed"
                details = {"hook_id": hook.hook_id, **dict(result.metadata)}
            elif result.metrics and not policy.allow_metric_emission:
                code = "learning_hook_metric_policy_violation"
                details = {"hook_id": hook.hook_id}
                result = None
            else:
                code = None
        if code:
            issue = LearningIssue(code, "hook dispatch failure", details)
            disable = policy.failure_mode == "disable_hook"
            receipts.append(
                HookExecutionReceipt(
                    hook.hook_id,
                    context.event_type,
                    context.event_sequence,
                    "fail",
                    disabled_after_failure=disable,
                    failure_code=code,
                    metadata=details,
                )
            )
            if policy.failure_mode == "fail_fast":
                blockers.append(issue)
                break
            warnings.extend(result.warnings if result is not None else ())
            warnings.append(
                LearningIssue(
                    "learning_hook_disabled"
                    if disable
                    else (
                        code
                        if code
                        in (
                            "learning_hook_result_invalid",
                            "learning_hook_metric_policy_violation",
                        )
                        else "learning_hook_failed_continue"
                    ),
                    issue.message,
                    issue.details,
                )
            )
            disabled.update((hook.hook_id,) if disable else ())
        else:
            receipts.append(
                HookExecutionReceipt(
                    hook.hook_id,
                    context.event_type,
                    context.event_sequence,
                    result.status,
                    len(result.metrics),
                    tuple(x.code for x in result.warnings),
                )
            )
            metrics.extend(result.metrics)
            warnings.extend(result.warnings)
    return HookDispatchResult(
        "fail" if blockers else "warning" if warnings else "pass",
        tuple(receipts),
        tuple(metrics),
        tuple(warnings),
        tuple(blockers),
        tuple(sorted(disabled)),
    )


def merge_core_and_hook_issues(core_blockers, result):
    return tuple(core_blockers) + result.blockers
