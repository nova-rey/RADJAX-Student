"""Immutable strict contracts for deterministic replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)

REPLAY_SCHEMA_VERSION = "radjax.p3_11_9_replay_evidence.v1"
_HEX_DIGEST_LENGTH = 64


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{name} must contain nonempty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def _strict(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ReplayCanonicalError(
            f"{name} fields differ; missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True)
class ReplayBlocker:
    code: str
    field: str
    expected_digest: str | None = None
    observed_digest: str | None = None
    mode: str | None = None
    arm: str | None = None
    step_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.startswith("replay_"):
            raise ValueError("replay blocker code must be stable")
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("replay blocker field must be nonempty")
        for name in ("expected_digest", "observed_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if self.mode is not None and self.mode not in {"eager", "jit"}:
            raise ValueError("replay blocker mode is invalid")
        if self.arm is not None and self.arm not in {"uninterrupted", "resumed"}:
            raise ValueError("replay blocker arm is invalid")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("replay blocker step index must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "expected_digest": self.expected_digest,
            "observed_digest": self.observed_digest,
            "mode": self.mode,
            "arm": self.arm,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class ReplayStepEvidence:
    step_index: int
    batch_id: str
    batch_digest: str
    objective_id: str
    objective_surface_id: str
    update_scope_digest: str
    counters_before: Mapping[str, int]
    counters_after: Mapping[str, int]
    parameter_digest: str
    architecture_carry_digest: str
    optimizer_array_digest: str
    optimizer_envelope_digest: str
    changed_paths: tuple[str, ...]
    unchanged_paths: tuple[str, ...]
    objective_metrics: Mapping[str, str]
    architecture_metrics: Mapping[str, str]
    optimizer_metrics: Mapping[str, str]
    hook_events: tuple[str, ...]
    runtime: Mapping[str, Any]
    rng: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be nonnegative")
        for name in ("batch_id", "objective_id", "objective_surface_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        for name in (
            "batch_digest",
            "update_scope_digest",
            "parameter_digest",
            "architecture_carry_digest",
            "optimizer_array_digest",
            "optimizer_envelope_digest",
        ):
            _digest(getattr(self, name), name)
        for name in ("counters_before", "counters_after"):
            counters = _mapping(getattr(self, name), name)
            if set(counters) != {"global_step", "micro_step", "optimizer_step"} or any(
                not isinstance(item, int) or item < 0 for item in counters.values()
            ):
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, counters)
        for name in ("changed_paths", "unchanged_paths", "hook_events"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if set(self.changed_paths) & set(self.unchanged_paths):
            raise ValueError("changed and unchanged paths overlap")
        for name in ("objective_metrics", "architecture_metrics", "optimizer_metrics"):
            values = _mapping(getattr(self, name), name)
            if any(
                not isinstance(key, str)
                or not key
                or not isinstance(item, str)
                or not item.startswith("0x")
                for key, item in values.items()
            ):
                raise ValueError(f"{name} must contain canonical float encodings")
            object.__setattr__(self, name, values)
        runtime = _mapping(self.runtime, "runtime")
        rng = _mapping(self.rng, "rng")
        if set(rng) != {
            "schema_version",
            "prng_implementation",
            "stream",
            "slot",
            "global_step",
            "micro_step",
            "invocation_index",
        }:
            raise ValueError("RNG evidence fields are invalid")
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "rng", rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "batch_id": self.batch_id,
            "batch_digest": self.batch_digest,
            "objective_id": self.objective_id,
            "objective_surface_id": self.objective_surface_id,
            "update_scope_digest": self.update_scope_digest,
            "counters_before": dict(self.counters_before),
            "counters_after": dict(self.counters_after),
            "parameter_digest": self.parameter_digest,
            "architecture_carry_digest": self.architecture_carry_digest,
            "optimizer_array_digest": self.optimizer_array_digest,
            "optimizer_envelope_digest": self.optimizer_envelope_digest,
            "changed_paths": list(self.changed_paths),
            "unchanged_paths": list(self.unchanged_paths),
            "objective_metrics": dict(self.objective_metrics),
            "architecture_metrics": dict(self.architecture_metrics),
            "optimizer_metrics": dict(self.optimizer_metrics),
            "hook_events": list(self.hook_events),
            "runtime": dict(self.runtime),
            "rng": dict(self.rng),
        }


@dataclass(frozen=True)
class ReplayArmEvidence:
    arm: str
    experiment_identity: Mapping[str, Any]
    lifecycle_identity: Mapping[str, Any]
    batch_sequence_digest: str
    checkpoint_boundary: int
    checkpoint_manifest_digest: str
    restore_used_caller_identity: bool
    steps: tuple[ReplayStepEvidence, ...]
    final_parameter_digest: str
    final_architecture_carry_digest: str
    final_optimizer_array_digest: str
    final_optimizer_envelope_digest: str
    final_learning_state_digest: str
    final_hook_digest: str
    retained_metrics_digest: str
    final_report_digest: str
    final_runtime: Mapping[str, Any]
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.arm not in {"uninterrupted", "resumed"}:
            raise ValueError("replay arm is invalid")
        object.__setattr__(
            self,
            "experiment_identity",
            _mapping(self.experiment_identity, "experiment_identity"),
        )
        object.__setattr__(
            self,
            "lifecycle_identity",
            _mapping(self.lifecycle_identity, "lifecycle_identity"),
        )
        for name in (
            "batch_sequence_digest",
            "checkpoint_manifest_digest",
            "final_parameter_digest",
            "final_architecture_carry_digest",
            "final_optimizer_array_digest",
            "final_optimizer_envelope_digest",
            "final_learning_state_digest",
            "final_hook_digest",
            "retained_metrics_digest",
            "final_report_digest",
        ):
            _digest(getattr(self, name), name)
        if self.checkpoint_boundary < 0 or not isinstance(
            self.restore_used_caller_identity, bool
        ):
            raise ValueError("replay checkpoint evidence is invalid")
        steps = tuple(self.steps)
        if not steps or tuple(step.step_index for step in steps) != tuple(
            range(len(steps))
        ):
            raise ValueError("replay steps must be ordered and contiguous")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self, "final_runtime", _mapping(self.final_runtime, "final_runtime")
        )
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "experiment_identity": dict(self.experiment_identity),
            "lifecycle_identity": dict(self.lifecycle_identity),
            "batch_sequence_digest": self.batch_sequence_digest,
            "checkpoint_boundary": self.checkpoint_boundary,
            "checkpoint_manifest_digest": self.checkpoint_manifest_digest,
            "restore_used_caller_identity": self.restore_used_caller_identity,
            "steps": [step.to_dict() for step in self.steps],
            "final_parameter_digest": self.final_parameter_digest,
            "final_architecture_carry_digest": self.final_architecture_carry_digest,
            "final_optimizer_array_digest": self.final_optimizer_array_digest,
            "final_optimizer_envelope_digest": self.final_optimizer_envelope_digest,
            "final_learning_state_digest": self.final_learning_state_digest,
            "final_hook_digest": self.final_hook_digest,
            "retained_metrics_digest": self.retained_metrics_digest,
            "final_report_digest": self.final_report_digest,
            "final_runtime": dict(self.final_runtime),
            "non_claims": list(self.non_claims),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class ReplayModeEvidence:
    mode: str
    replay_a: ReplayArmEvidence
    replay_b: ReplayArmEvidence

    def __post_init__(self) -> None:
        if self.mode not in {"eager", "jit"}:
            raise ValueError("replay mode is invalid")
        if self.replay_a.arm != "uninterrupted" or self.replay_b.arm != "uninterrupted":
            raise ValueError("mode replay anchors must be uninterrupted arms")


@dataclass(frozen=True)
class ReplayRunEvidence:
    """The uninterrupted and resumed arms from one fresh experiment run."""

    uninterrupted: ReplayArmEvidence
    resumed: ReplayArmEvidence

    def __post_init__(self) -> None:
        if self.uninterrupted.arm != "uninterrupted" or self.resumed.arm != "resumed":
            raise ValueError("replay run arm labels are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uninterrupted": self.uninterrupted.to_dict(),
            "resumed": self.resumed.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class StatefulReplayProof:
    experiment_identity: Mapping[str, Any]
    replay_count: int
    tolerance: Mapping[str, str]
    modes: Mapping[str, Mapping[str, ReplayRunEvidence]]
    cross_mode: Mapping[str, Any]
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_identity",
            _mapping(self.experiment_identity, "experiment_identity"),
        )
        if self.replay_count != 2:
            raise ValueError("P3.11.9 requires exactly two replays")
        tolerance = _mapping(self.tolerance, "tolerance")
        if set(tolerance) != {"rtol", "atol"} or any(
            not isinstance(value, str) or not value.startswith("0x")
            for value in tolerance.values()
        ):
            raise ValueError("replay tolerance must use float hex")
        if set(self.modes) != {"eager", "jit"}:
            raise ValueError("replay proof requires eager and jit modes")
        frozen_modes: dict[str, Mapping[str, ReplayRunEvidence]] = {}
        for mode, replays in self.modes.items():
            if set(replays) != {"replay_a", "replay_b"}:
                raise ValueError("mode evidence requires replay_a and replay_b")
            for replay in replays.values():
                if not isinstance(replay, ReplayRunEvidence):
                    raise TypeError("mode replay evidence is invalid")
            frozen_modes[mode] = MappingProxyType(dict(replays))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "modes", MappingProxyType(frozen_modes))
        object.__setattr__(self, "cross_mode", _mapping(self.cross_mode, "cross_mode"))
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))


@dataclass(frozen=True)
class ReplayVerificationResult:
    blockers: tuple[ReplayBlocker, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "blockers": [item.to_dict() for item in self.blockers],
        }


@dataclass(frozen=True)
class StatefulReplayReceipt:
    proof: StatefulReplayProof
    verification: ReplayVerificationResult
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay receipt schema")
        if self.verification.passed is False:
            raise ValueError("a passing replay artifact requires verifier success")

    def to_dict(self) -> dict[str, Any]:
        modes: dict[str, Any] = {}
        for mode in ("eager", "jit"):
            replay_a = self.proof.modes[mode]["replay_a"]
            replay_b = self.proof.modes[mode]["replay_b"]
            modes[mode] = {
                "canonical_trace": replay_a.uninterrupted.to_dict(),
                "replay_a_digest": replay_a.digest,
                "replay_b_digest": replay_b.digest,
                "uninterrupted_arm_digest": replay_a.uninterrupted.digest,
                "resumed_arm_digest": replay_a.resumed.digest,
            }
        payload = {
            "schema_version": self.schema_version,
            "status": "pass",
            "experiment_identity": dict(self.proof.experiment_identity),
            "replay_count": self.proof.replay_count,
            "execution_modes": ["eager", "jit"],
            "cross_mode_tolerance": dict(self.proof.tolerance),
            "modes": modes,
            "cross_mode": dict(self.proof.cross_mode),
            "verifier": self.verification.to_dict(),
            "non_claims": list(self.proof.non_claims),
        }
        payload["evidence_digest"] = canonical_digest(payload)
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes | str) -> Mapping[str, Any]:
        payload = parse_canonical_json(data)
        if not isinstance(payload, Mapping):
            raise ReplayCanonicalError("replay receipt must be an object")
        _strict(
            payload,
            {
                "schema_version",
                "status",
                "experiment_identity",
                "replay_count",
                "execution_modes",
                "cross_mode_tolerance",
                "modes",
                "cross_mode",
                "verifier",
                "non_claims",
                "evidence_digest",
            },
            "replay receipt",
        )
        if (
            payload["schema_version"] != REPLAY_SCHEMA_VERSION
            or payload["status"] != "pass"
        ):
            raise ReplayCanonicalError("replay receipt schema or status is invalid")
        if payload["evidence_digest"] != canonical_digest(
            {key: value for key, value in payload.items() if key != "evidence_digest"}
        ):
            raise ReplayCanonicalError("replay receipt evidence digest mismatch")
        if payload["execution_modes"] != ["eager", "jit"] or set(payload["modes"]) != {
            "eager",
            "jit",
        }:
            raise ReplayCanonicalError("replay receipt modes are invalid")
        mode_fields = {
            "canonical_trace",
            "replay_a_digest",
            "replay_b_digest",
            "uninterrupted_arm_digest",
            "resumed_arm_digest",
        }
        arm_fields = {
            "arm",
            "experiment_identity",
            "lifecycle_identity",
            "batch_sequence_digest",
            "checkpoint_boundary",
            "checkpoint_manifest_digest",
            "restore_used_caller_identity",
            "steps",
            "final_parameter_digest",
            "final_architecture_carry_digest",
            "final_optimizer_array_digest",
            "final_optimizer_envelope_digest",
            "final_learning_state_digest",
            "final_hook_digest",
            "retained_metrics_digest",
            "final_report_digest",
            "final_runtime",
            "non_claims",
        }
        step_fields = {
            "step_index",
            "batch_id",
            "batch_digest",
            "objective_id",
            "objective_surface_id",
            "update_scope_digest",
            "counters_before",
            "counters_after",
            "parameter_digest",
            "architecture_carry_digest",
            "optimizer_array_digest",
            "optimizer_envelope_digest",
            "changed_paths",
            "unchanged_paths",
            "objective_metrics",
            "architecture_metrics",
            "optimizer_metrics",
            "hook_events",
            "runtime",
            "rng",
        }
        for mode in ("eager", "jit"):
            mode_payload = payload["modes"][mode]
            if not isinstance(mode_payload, Mapping):
                raise ReplayCanonicalError("replay mode must be an object")
            _strict(mode_payload, mode_fields, "replay mode")
            trace = mode_payload["canonical_trace"]
            if not isinstance(trace, Mapping):
                raise ReplayCanonicalError("replay trace must be an object")
            _strict(trace, arm_fields, "replay arm")
            if not isinstance(trace["steps"], list):
                raise ReplayCanonicalError("replay trace steps must be a list")
            for step in trace["steps"]:
                if not isinstance(step, Mapping):
                    raise ReplayCanonicalError("replay step must be an object")
                _strict(step, step_fields, "replay step")
        return payload


__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "ReplayArmEvidence",
    "ReplayBlocker",
    "ReplayModeEvidence",
    "ReplayRunEvidence",
    "ReplayStepEvidence",
    "ReplayVerificationResult",
    "StatefulReplayProof",
    "StatefulReplayReceipt",
]
