"""Observer-only deterministic lifecycle hooks; not loop integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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


@dataclass(frozen=True)
class HookContext:
    run_id: str
    event_type: str
    event_sequence: int
    global_step: int
    metrics: tuple[MetricRecord, ...] = ()

    def __post_init__(self):
        if (
            self.event_type not in HOOK_EVENTS
            or self.event_sequence < 0
            or self.global_step < 0
        ):
            raise ValueError("hook context is invalid")


@dataclass(frozen=True)
class HookResult:
    status: Literal["pass", "warning", "fail"] = "pass"
    metrics: tuple[MetricRecord, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()


class LearningHook(Protocol):
    hook_id: str
    priority: int

    def on_event(self, context: HookContext) -> HookResult: ...


@dataclass(frozen=True)
class HookPolicy:
    failure_mode: Literal["fail_fast", "warn_and_continue", "disable_hook"] = (
        "fail_fast"
    )


@dataclass(frozen=True)
class HookExecutionReceipt:
    hook_id: str
    event_type: str
    event_sequence: int
    status: str
    disabled_after_failure: bool = False


@dataclass(frozen=True)
class HookDispatchResult:
    receipts: tuple[HookExecutionReceipt, ...]
    metrics: tuple[MetricRecord, ...]
    warnings: tuple[LearningIssue, ...]
    blockers: tuple[LearningIssue, ...]
    disabled_hook_ids: tuple[str, ...]


def dispatch_hooks(
    hooks, policy: HookPolicy, context: HookContext, disabled_hook_ids=()
):
    disabled = set(disabled_hook_ids)
    ordered = sorted(hooks, key=lambda h: (h.priority, h.hook_id))
    if len({h.hook_id for h in ordered}) != len(ordered):
        raise ValueError("learning_hook_duplicate")
    receipts = []
    metrics = []
    warnings = []
    blockers = []
    for hook in ordered:
        if hook.hook_id in disabled:
            continue
        try:
            result = hook.on_event(context)
        except Exception:
            result = HookResult(
                status="fail",
                warnings=(
                    LearningIssue(
                        "learning_hook_failed", "hook raised", {"hook_id": hook.hook_id}
                    ),
                ),
            )
        if not isinstance(result, HookResult):
            result = HookResult(
                status="fail",
                warnings=(
                    LearningIssue(
                        "learning_hook_result_invalid",
                        "hook returned invalid result",
                        {"hook_id": hook.hook_id},
                    ),
                ),
            )
        failed = result.status == "fail"
        disable = failed and policy.failure_mode == "disable_hook"
        receipts.append(
            HookExecutionReceipt(
                hook.hook_id,
                context.event_type,
                context.event_sequence,
                result.status,
                disable,
            )
        )
        if failed:
            issue = LearningIssue(
                "learning_hook_failed", "hook failed", {"hook_id": hook.hook_id}
            )
            if policy.failure_mode == "fail_fast":
                blockers.append(issue)
                break
            warnings.append(
                LearningIssue(
                    "learning_hook_disabled"
                    if disable
                    else "learning_hook_failed_continue",
                    issue.message,
                    issue.details,
                )
            )
            if disable:
                disabled.add(hook.hook_id)
        else:
            metrics.extend(result.metrics)
            warnings.extend(result.warnings)
    return HookDispatchResult(
        tuple(receipts),
        tuple(metrics),
        tuple(warnings),
        tuple(blockers),
        tuple(sorted(disabled)),
    )
