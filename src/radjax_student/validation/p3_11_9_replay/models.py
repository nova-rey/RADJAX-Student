"""Immutable, recursively strict contracts for P3.11.9 replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
    parse_finite_float_hex,
)

REPLAY_SCHEMA_VERSION = "radjax.p3_11_9_replay_evidence.v1"
_HEX_DIGEST_LENGTH = 64


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayCanonicalError(f"{name} must be an object")
    return MappingProxyType(dict(value))


def _strict(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ReplayCanonicalError(
            f"{name} fields differ; missing={missing}, unknown={unknown}"
        )


def _string(value: Any, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ReplayCanonicalError(f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayCanonicalError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReplayCanonicalError(f"{name} must be a nonnegative integer")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReplayCanonicalError(f"{name} must be a list of nonempty strings")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ReplayCanonicalError(f"{name} contains duplicates")
    return result


def _freeze_json(value: Any, name: str = "metadata") -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ReplayCanonicalError(f"{name} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item, name) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name) for item in value)
    raise ReplayCanonicalError(f"{name} must contain JSON-safe values")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _metric_mapping(value: Any, name: str) -> Mapping[str, str]:
    payload = _mapping(value, name)
    result: dict[str, str] = {}
    for key, item in payload.items():
        result[_string(key, f"{name} key")] = item
        parse_finite_float_hex(item)
    return MappingProxyType({key: result[key] for key in sorted(result)})


@dataclass(frozen=True)
class ArchitectureCarryIdentityEvidence:
    schema_version: str
    state_id: str | None
    pytree_descriptor_digest: str

    _FIELDS = {"schema_version", "state_id", "pytree_descriptor_digest"}

    def __post_init__(self) -> None:
        if self.schema_version != "architecture_carry.v1":
            raise ReplayCanonicalError("unsupported architecture carry schema")
        object.__setattr__(
            self, "state_id", _string(self.state_id, "state_id", allow_none=True)
        )
        _digest(self.pytree_descriptor_digest, "pytree_descriptor_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": self.state_id,
            "pytree_descriptor_digest": self.pytree_descriptor_digest,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArchitectureCarryIdentityEvidence:
        payload = _mapping(value, "architecture_carry_descriptor")
        _strict(payload, cls._FIELDS, "architecture_carry_descriptor")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFPreservationEvidence:
    descriptor_schema_version: str
    descriptor_digest: str
    model_type: str
    architecture_id: str
    tokenizer_id: str
    vocabulary_size: int
    special_token_digest: str
    parameter_layout_digest: str
    architecture_config_digest: str

    _FIELDS = {
        "descriptor_schema_version",
        "descriptor_digest",
        "model_type",
        "architecture_id",
        "tokenizer_id",
        "vocabulary_size",
        "special_token_digest",
        "parameter_layout_digest",
        "architecture_config_digest",
    }

    def __post_init__(self) -> None:
        for name in (
            "descriptor_schema_version",
            "descriptor_digest",
            "model_type",
            "architecture_id",
            "tokenizer_id",
            "special_token_digest",
        ):
            _string(getattr(self, name), name)
        if not isinstance(self.vocabulary_size, int) or self.vocabulary_size <= 0:
            raise ReplayCanonicalError("vocabulary_size must be a positive integer")
        for name in ("parameter_layout_digest", "architecture_config_digest"):
            _digest(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_digest": self.descriptor_digest,
            "model_type": self.model_type,
            "architecture_id": self.architecture_id,
            "tokenizer_id": self.tokenizer_id,
            "vocabulary_size": self.vocabulary_size,
            "special_token_digest": self.special_token_digest,
            "parameter_layout_digest": self.parameter_layout_digest,
            "architecture_config_digest": self.architecture_config_digest,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HFPreservationEvidence:
        payload = _mapping(value, "hf_reference")
        _strict(payload, cls._FIELDS, "hf_reference")
        return cls(**dict(payload))


@dataclass(frozen=True)
class OptimizerConfigEvidence:
    optimizer_id: str
    schema_version: str
    learning_rate: float
    weight_decay: float
    weight_decay_mode: str
    gradient_clip_mode: str
    gradient_clip: float | None
    epsilon: float | None
    momentum: float | None
    schedule_reference: str | None
    metadata: Mapping[str, Any]

    _FIELDS = {
        "optimizer_id",
        "schema_version",
        "learning_rate",
        "weight_decay",
        "weight_decay_mode",
        "gradient_clip_mode",
        "gradient_clip",
        "epsilon",
        "momentum",
        "schedule_reference",
        "metadata",
    }

    def __post_init__(self) -> None:
        import math

        for name in (
            "optimizer_id",
            "schema_version",
            "weight_decay_mode",
            "gradient_clip_mode",
        ):
            _string(getattr(self, name), name)
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ReplayCanonicalError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ReplayCanonicalError("optimizer config numeric values are invalid")
        for name in ("gradient_clip", "epsilon", "momentum"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ReplayCanonicalError(f"{name} must be finite or null")
        object.__setattr__(
            self,
            "schedule_reference",
            _string(self.schedule_reference, "schedule_reference", allow_none=True),
        )
        metadata = _mapping(self.metadata, "optimizer metadata")
        object.__setattr__(
            self, "metadata", _freeze_json(metadata, "optimizer metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_id": self.optimizer_id,
            "schema_version": self.schema_version,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "weight_decay_mode": self.weight_decay_mode,
            "gradient_clip_mode": self.gradient_clip_mode,
            "gradient_clip": self.gradient_clip,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "schedule_reference": self.schedule_reference,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> OptimizerConfigEvidence:
        payload = _mapping(value, "optimizer_config")
        _strict(payload, cls._FIELDS, "optimizer_config")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ExperimentIdentityEvidence:
    architecture_id: str
    architecture_config_digest: str
    parameter_catalog_digest: str
    parameter_layout_digest: str
    hf_reference: HFPreservationEvidence
    architecture_state_id: str | None
    architecture_carry_descriptor: ArchitectureCarryIdentityEvidence
    optimizer_id: str
    optimizer_capability_version: int
    optimizer_numerical_state_schema_version: str
    optimizer_config: OptimizerConfigEvidence
    runtime_reference: str
    root_seed: int

    _FIELDS = {
        "architecture_id",
        "architecture_config_digest",
        "parameter_catalog_digest",
        "parameter_layout_digest",
        "hf_reference",
        "architecture_state_id",
        "architecture_carry_descriptor",
        "optimizer_id",
        "optimizer_capability_version",
        "optimizer_numerical_state_schema_version",
        "optimizer_config",
        "runtime_reference",
        "root_seed",
    }

    def __post_init__(self) -> None:
        _string(self.architecture_id, "architecture_id")
        for name in (
            "architecture_config_digest",
            "parameter_catalog_digest",
            "parameter_layout_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.hf_reference, HFPreservationEvidence):
            raise ReplayCanonicalError("hf_reference is invalid")
        object.__setattr__(
            self,
            "architecture_state_id",
            _string(
                self.architecture_state_id, "architecture_state_id", allow_none=True
            ),
        )
        if not isinstance(
            self.architecture_carry_descriptor, ArchitectureCarryIdentityEvidence
        ):
            raise ReplayCanonicalError("architecture_carry_descriptor is invalid")
        _string(self.optimizer_id, "optimizer_id")
        if (
            not isinstance(self.optimizer_capability_version, int)
            or isinstance(self.optimizer_capability_version, bool)
            or self.optimizer_capability_version <= 0
        ):
            raise ReplayCanonicalError("optimizer_capability_version must be positive")
        _string(
            self.optimizer_numerical_state_schema_version,
            "optimizer_numerical_state_schema_version",
        )
        if not isinstance(self.optimizer_config, OptimizerConfigEvidence):
            raise ReplayCanonicalError("optimizer_config is invalid")
        if self.optimizer_config.optimizer_id != self.optimizer_id:
            raise ReplayCanonicalError(
                "optimizer config identity differs from experiment"
            )
        _string(self.runtime_reference, "runtime_reference")
        _nonnegative_int(self.root_seed, "root_seed")
        if self.hf_reference.architecture_id != self.architecture_id:
            raise ReplayCanonicalError(
                "HF architecture identity differs from experiment"
            )
        if self.hf_reference.parameter_layout_digest != self.parameter_layout_digest:
            raise ReplayCanonicalError("HF layout identity differs from experiment")
        if (
            self.hf_reference.architecture_config_digest
            != self.architecture_config_digest
        ):
            raise ReplayCanonicalError("HF config identity differs from experiment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "architecture_config_digest": self.architecture_config_digest,
            "parameter_catalog_digest": self.parameter_catalog_digest,
            "parameter_layout_digest": self.parameter_layout_digest,
            "hf_reference": self.hf_reference.to_dict(),
            "architecture_state_id": self.architecture_state_id,
            "architecture_carry_descriptor": (
                self.architecture_carry_descriptor.to_dict()
            ),
            "optimizer_id": self.optimizer_id,
            "optimizer_capability_version": self.optimizer_capability_version,
            "optimizer_numerical_state_schema_version": (
                self.optimizer_numerical_state_schema_version
            ),
            "optimizer_config": self.optimizer_config.to_dict(),
            "runtime_reference": self.runtime_reference,
            "root_seed": self.root_seed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExperimentIdentityEvidence:
        payload = _mapping(value, "experiment_identity")
        _strict(payload, cls._FIELDS, "experiment_identity")
        values = dict(payload)
        values["hf_reference"] = HFPreservationEvidence.from_dict(
            values["hf_reference"]
        )
        values["architecture_carry_descriptor"] = (
            ArchitectureCarryIdentityEvidence.from_dict(
                values["architecture_carry_descriptor"]
            )
        )
        values["optimizer_config"] = OptimizerConfigEvidence.from_dict(
            values["optimizer_config"]
        )
        return cls(**values)


@dataclass(frozen=True)
class RuntimeEvidence:
    backend_id: str
    mode: str
    compiled: bool
    dispatched: bool
    synchronized: bool
    placement_policy: str
    precision_policy: str
    output_metadata_fields: tuple[str, ...]
    non_claims: tuple[str, ...]

    _FIELDS = {
        "backend_id",
        "mode",
        "compiled",
        "dispatched",
        "synchronized",
        "placement_policy",
        "precision_policy",
        "output_metadata_fields",
        "non_claims",
    }
    _FORBIDDEN_METADATA = {
        "selected_device_id",
        "device_serial",
        "duration",
        "elapsed",
        "temporary_path",
        "temp_path",
        "timestamp",
    }

    def __post_init__(self) -> None:
        for name in ("backend_id", "placement_policy", "precision_policy"):
            _string(getattr(self, name), name)
        if self.mode not in {"eager", "jit"}:
            raise ReplayCanonicalError("runtime mode is invalid")
        for name in ("compiled", "dispatched", "synchronized"):
            if not isinstance(getattr(self, name), bool):
                raise ReplayCanonicalError(f"{name} must be boolean")
        fields = _strings(self.output_metadata_fields, "output_metadata_fields")
        if tuple(sorted(fields)) != fields or set(fields) & self._FORBIDDEN_METADATA:
            raise ReplayCanonicalError("runtime metadata field set is invalid")
        object.__setattr__(self, "output_metadata_fields", fields)
        object.__setattr__(
            self, "non_claims", _strings(self.non_claims, "runtime non_claims")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "mode": self.mode,
            "compiled": self.compiled,
            "dispatched": self.dispatched,
            "synchronized": self.synchronized,
            "placement_policy": self.placement_policy,
            "precision_policy": self.precision_policy,
            "output_metadata_fields": list(self.output_metadata_fields),
            "non_claims": list(self.non_claims),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RuntimeEvidence:
        payload = _mapping(value, "runtime")
        _strict(payload, cls._FIELDS, "runtime")
        return cls(**dict(payload))


@dataclass(frozen=True)
class RngEvidence:
    schema_version: str
    prng_implementation: str
    stream: str
    slot: str
    global_step: int
    micro_step: int
    invocation_index: int

    _FIELDS = {
        "schema_version",
        "prng_implementation",
        "stream",
        "slot",
        "global_step",
        "micro_step",
        "invocation_index",
    }

    def __post_init__(self) -> None:
        for name in ("schema_version", "prng_implementation", "stream", "slot"):
            _string(getattr(self, name), name)
        for name in ("global_step", "micro_step", "invocation_index"):
            _nonnegative_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prng_implementation": self.prng_implementation,
            "stream": self.stream,
            "slot": self.slot,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "invocation_index": self.invocation_index,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RngEvidence:
        payload = _mapping(value, "rng")
        _strict(payload, cls._FIELDS, "rng")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ToleranceEvidence:
    rtol: str
    atol: str

    _FIELDS = {"rtol", "atol"}

    def __post_init__(self) -> None:
        parse_finite_float_hex(self.rtol, positive=True)
        parse_finite_float_hex(self.atol, positive=True)

    def to_dict(self) -> dict[str, str]:
        return {"rtol": self.rtol, "atol": self.atol}

    @classmethod
    def from_dict(cls, value: Any) -> ToleranceEvidence:
        payload = _mapping(value, "tolerance")
        _strict(payload, cls._FIELDS, "tolerance")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CrossModeComparisonEvidence:
    structure_equal: bool
    keypaths_equal: bool
    leaf_count_equal: bool
    dtype_shape_equal: bool
    integer_values_equal: bool
    floating_values_within_tolerance: bool
    learning_state_equal: bool
    optimizer_envelope_equal: bool
    lifecycle_identity_equal: bool
    hook_sequence_equal: bool
    metric_names_equal: bool
    metric_values_within_tolerance: bool
    logical_paths_equal: bool
    rng_identity_equal: bool
    runtime_structure_equal: bool
    declared_rtol: str
    declared_atol: str

    _FIELDS = {
        "structure_equal",
        "keypaths_equal",
        "leaf_count_equal",
        "dtype_shape_equal",
        "integer_values_equal",
        "floating_values_within_tolerance",
        "learning_state_equal",
        "optimizer_envelope_equal",
        "lifecycle_identity_equal",
        "hook_sequence_equal",
        "metric_names_equal",
        "metric_values_within_tolerance",
        "logical_paths_equal",
        "rng_identity_equal",
        "runtime_structure_equal",
        "declared_rtol",
        "declared_atol",
    }
    _BOOLEAN_FIELDS = _FIELDS - {"declared_rtol", "declared_atol"}

    def __post_init__(self) -> None:
        for name in self._BOOLEAN_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise ReplayCanonicalError(f"{name} must be boolean")
        parse_finite_float_hex(self.declared_rtol, positive=True)
        parse_finite_float_hex(self.declared_atol, positive=True)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, value: Any) -> CrossModeComparisonEvidence:
        payload = _mapping(value, "cross_mode")
        _strict(payload, cls._FIELDS, "cross_mode")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ReplayBlocker:
    code: str
    field: str
    expected_digest: str | None = None
    observed_digest: str | None = None
    mode: str | None = None
    arm: str | None = None
    step_index: int | None = None

    _FIELDS = {
        "code",
        "field",
        "expected_digest",
        "observed_digest",
        "mode",
        "arm",
        "step_index",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.startswith("replay_"):
            raise ReplayCanonicalError("replay blocker code must be stable")
        _string(self.field, "replay blocker field")
        for name in ("expected_digest", "observed_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if self.mode is not None and self.mode not in {"eager", "jit"}:
            raise ReplayCanonicalError("replay blocker mode is invalid")
        if self.arm is not None and self.arm not in {"uninterrupted", "resumed"}:
            raise ReplayCanonicalError("replay blocker arm is invalid")
        if self.step_index is not None:
            _nonnegative_int(self.step_index, "replay blocker step_index")

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

    @classmethod
    def from_dict(cls, value: Any) -> ReplayBlocker:
        payload = _mapping(value, "replay blocker")
        _strict(payload, cls._FIELDS, "replay blocker")
        return cls(**dict(payload))


@dataclass(frozen=True)
class VerifierEvidence:
    status: str
    blockers: tuple[ReplayBlocker, ...]

    _FIELDS = {"status", "blockers"}

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail"}:
            raise ReplayCanonicalError("verifier status is invalid")
        if not isinstance(self.blockers, (list, tuple)) or not all(
            isinstance(item, ReplayBlocker) for item in self.blockers
        ):
            raise ReplayCanonicalError("verifier blockers are invalid")
        blockers = tuple(self.blockers)
        if (self.status == "pass") != (not blockers):
            raise ReplayCanonicalError("verifier status and blockers disagree")
        object.__setattr__(self, "blockers", blockers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
        }

    @classmethod
    def from_dict(cls, value: Any) -> VerifierEvidence:
        payload = _mapping(value, "verifier")
        _strict(payload, cls._FIELDS, "verifier")
        blockers = payload["blockers"]
        if not isinstance(blockers, list):
            raise ReplayCanonicalError("verifier blockers must be a list")
        return cls(
            payload["status"], tuple(ReplayBlocker.from_dict(item) for item in blockers)
        )


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
    runtime: RuntimeEvidence
    rng: RngEvidence

    _FIELDS = {
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

    def __post_init__(self) -> None:
        _nonnegative_int(self.step_index, "step_index")
        for name in ("batch_id", "objective_id", "objective_surface_id"):
            _string(getattr(self, name), name)
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
            _strict(counters, {"global_step", "micro_step", "optimizer_step"}, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        key: _nonnegative_int(value, f"{name}.{key}")
                        for key, value in counters.items()
                    }
                ),
            )
        for name in ("changed_paths", "unchanged_paths", "hook_events"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if set(self.changed_paths) & set(self.unchanged_paths):
            raise ReplayCanonicalError("changed and unchanged paths overlap")
        for name in ("objective_metrics", "architecture_metrics", "optimizer_metrics"):
            object.__setattr__(self, name, _metric_mapping(getattr(self, name), name))
        if not isinstance(self.runtime, RuntimeEvidence) or not isinstance(
            self.rng, RngEvidence
        ):
            raise ReplayCanonicalError("step runtime or RNG evidence is invalid")

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
            "runtime": self.runtime.to_dict(),
            "rng": self.rng.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReplayStepEvidence:
        payload = _mapping(value, "replay step")
        _strict(payload, cls._FIELDS, "replay step")
        values = dict(payload)
        values["runtime"] = RuntimeEvidence.from_dict(values["runtime"])
        values["rng"] = RngEvidence.from_dict(values["rng"])
        return cls(**values)


@dataclass(frozen=True)
class ReplayArmEvidence:
    arm: str
    experiment_identity: ExperimentIdentityEvidence
    lifecycle_identity: ExperimentIdentityEvidence
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
    final_runtime: RuntimeEvidence
    non_claims: tuple[str, ...]

    _FIELDS = {
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

    def __post_init__(self) -> None:
        if self.arm not in {"uninterrupted", "resumed"}:
            raise ReplayCanonicalError("replay arm is invalid")
        if not isinstance(
            self.experiment_identity, ExperimentIdentityEvidence
        ) or not isinstance(self.lifecycle_identity, ExperimentIdentityEvidence):
            raise ReplayCanonicalError("replay lifecycle evidence is invalid")
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
        _nonnegative_int(self.checkpoint_boundary, "checkpoint_boundary")
        if not isinstance(self.restore_used_caller_identity, bool):
            raise ReplayCanonicalError("restore_used_caller_identity must be boolean")
        if not isinstance(self.steps, (tuple, list)) or not all(
            isinstance(step, ReplayStepEvidence) for step in self.steps
        ):
            raise ReplayCanonicalError("replay steps are invalid")
        steps = tuple(self.steps)
        if not steps or tuple(step.step_index for step in steps) != tuple(
            range(len(steps))
        ):
            raise ReplayCanonicalError("replay steps must be ordered and contiguous")
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.final_runtime, RuntimeEvidence):
            raise ReplayCanonicalError("final runtime evidence is invalid")
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "experiment_identity": self.experiment_identity.to_dict(),
            "lifecycle_identity": self.lifecycle_identity.to_dict(),
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
            "final_runtime": self.final_runtime.to_dict(),
            "non_claims": list(self.non_claims),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReplayArmEvidence:
        payload = _mapping(value, "replay arm")
        _strict(payload, cls._FIELDS, "replay arm")
        values = dict(payload)
        values["experiment_identity"] = ExperimentIdentityEvidence.from_dict(
            values["experiment_identity"]
        )
        values["lifecycle_identity"] = ExperimentIdentityEvidence.from_dict(
            values["lifecycle_identity"]
        )
        if not isinstance(values["steps"], list):
            raise ReplayCanonicalError("replay arm steps must be a list")
        values["steps"] = tuple(
            ReplayStepEvidence.from_dict(item) for item in values["steps"]
        )
        values["final_runtime"] = RuntimeEvidence.from_dict(values["final_runtime"])
        return cls(**values)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class ReplayRunEvidence:
    uninterrupted: ReplayArmEvidence
    resumed: ReplayArmEvidence

    def __post_init__(self) -> None:
        if self.uninterrupted.arm != "uninterrupted" or self.resumed.arm != "resumed":
            raise ReplayCanonicalError("replay run arm labels are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uninterrupted": self.uninterrupted.to_dict(),
            "resumed": self.resumed.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class ReplayModeEvidence:
    """Compatibility envelope for the two uninterrupted mode anchors."""

    mode: str
    replay_a: ReplayArmEvidence
    replay_b: ReplayArmEvidence

    def __post_init__(self) -> None:
        if self.mode not in {"eager", "jit"}:
            raise ReplayCanonicalError("replay mode is invalid")
        if self.replay_a.arm != "uninterrupted" or self.replay_b.arm != "uninterrupted":
            raise ReplayCanonicalError("mode replay anchors must be uninterrupted")


@dataclass(frozen=True)
class ReplayVerificationResult:
    blockers: tuple[ReplayBlocker, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.blockers, (tuple, list)) or not all(
            isinstance(item, ReplayBlocker) for item in self.blockers
        ):
            raise ReplayCanonicalError("replay verification blockers are invalid")
        object.__setattr__(self, "blockers", tuple(self.blockers))

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return VerifierEvidence(
            "pass" if self.passed else "fail", self.blockers
        ).to_dict()


@dataclass(frozen=True)
class StatefulReplayProof:
    experiment_identity: ExperimentIdentityEvidence
    replay_count: int
    tolerance: ToleranceEvidence
    modes: Mapping[str, Mapping[str, ReplayRunEvidence]]
    cross_mode: CrossModeComparisonEvidence
    non_claims: tuple[str, ...]
    executed_cross_mode: CrossModeComparisonEvidence | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_identity, ExperimentIdentityEvidence):
            raise ReplayCanonicalError("experiment_identity is invalid")
        if self.replay_count != 2:
            raise ReplayCanonicalError("P3.11.9 requires exactly two replays")
        if not isinstance(self.tolerance, ToleranceEvidence):
            raise ReplayCanonicalError("replay tolerance is invalid")
        if set(self.modes) != {"eager", "jit"}:
            raise ReplayCanonicalError("replay proof requires eager and jit modes")
        frozen_modes: dict[str, Mapping[str, ReplayRunEvidence]] = {}
        for mode, replays in self.modes.items():
            if set(replays) != {"replay_a", "replay_b"} or not all(
                isinstance(replay, ReplayRunEvidence) for replay in replays.values()
            ):
                raise ReplayCanonicalError("mode replay evidence is invalid")
            frozen_modes[mode] = MappingProxyType(dict(replays))
        object.__setattr__(self, "modes", MappingProxyType(frozen_modes))
        if not isinstance(self.cross_mode, CrossModeComparisonEvidence):
            raise ReplayCanonicalError("cross-mode evidence is invalid")
        if (
            self.cross_mode.declared_rtol != self.tolerance.rtol
            or self.cross_mode.declared_atol != self.tolerance.atol
        ):
            raise ReplayCanonicalError(
                "cross-mode tolerance differs from declared tolerance"
            )
        if self.executed_cross_mode is not None and not isinstance(
            self.executed_cross_mode, CrossModeComparisonEvidence
        ):
            raise ReplayCanonicalError("executed cross-mode evidence is invalid")
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))


@dataclass(frozen=True)
class StatefulReplayReceipt:
    proof: StatefulReplayProof
    verification: ReplayVerificationResult
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplayCanonicalError("unsupported replay receipt schema")
        if not self.verification.passed:
            raise ReplayCanonicalError(
                "a passing replay artifact requires verifier success"
            )

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
            "experiment_identity": self.proof.experiment_identity.to_dict(),
            "replay_count": self.proof.replay_count,
            "execution_modes": ["eager", "jit"],
            "cross_mode_tolerance": self.proof.tolerance.to_dict(),
            "modes": modes,
            "cross_mode": self.proof.cross_mode.to_dict(),
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
        if not isinstance(payload["replay_count"], int) or payload["replay_count"] != 2:
            raise ReplayCanonicalError("replay receipt count is invalid")
        if payload["execution_modes"] != ["eager", "jit"]:
            raise ReplayCanonicalError("replay receipt execution modes are invalid")
        _digest(payload["evidence_digest"], "evidence_digest")
        evidence_payload = {
            key: value for key, value in payload.items() if key != "evidence_digest"
        }
        if payload["evidence_digest"] != canonical_digest(evidence_payload):
            raise ReplayCanonicalError("replay receipt evidence digest mismatch")
        experiment = ExperimentIdentityEvidence.from_dict(
            payload["experiment_identity"]
        )
        tolerance = ToleranceEvidence.from_dict(payload["cross_mode_tolerance"])
        cross_mode = CrossModeComparisonEvidence.from_dict(payload["cross_mode"])
        if (
            cross_mode.declared_rtol != tolerance.rtol
            or cross_mode.declared_atol != tolerance.atol
        ):
            raise ReplayCanonicalError("receipt cross-mode tolerance is inconsistent")
        verifier = VerifierEvidence.from_dict(payload["verifier"])
        if verifier.status != "pass" or verifier.blockers:
            raise ReplayCanonicalError(
                "a recorded replay artifact must have passing verifier evidence"
            )
        _strings(payload["non_claims"], "non_claims")
        modes = _mapping(payload["modes"], "replay modes")
        _strict(modes, {"eager", "jit"}, "replay modes")
        mode_fields = {
            "canonical_trace",
            "replay_a_digest",
            "replay_b_digest",
            "uninterrupted_arm_digest",
            "resumed_arm_digest",
        }
        for mode in ("eager", "jit"):
            mode_payload = _mapping(modes[mode], f"replay mode {mode}")
            _strict(mode_payload, mode_fields, f"replay mode {mode}")
            trace = ReplayArmEvidence.from_dict(mode_payload["canonical_trace"])
            if trace.arm != "uninterrupted":
                raise ReplayCanonicalError(
                    "canonical replay trace must be uninterrupted"
                )
            if trace.experiment_identity != experiment:
                raise ReplayCanonicalError("replay trace experiment identity differs")
            if trace.final_runtime.mode != mode:
                raise ReplayCanonicalError("replay trace runtime mode differs")
            for step in trace.steps:
                if step.runtime.mode != mode:
                    raise ReplayCanonicalError("replay step runtime mode differs")
            for name in (
                "replay_a_digest",
                "replay_b_digest",
                "uninterrupted_arm_digest",
                "resumed_arm_digest",
            ):
                _digest(mode_payload[name], f"{mode}.{name}")
            if mode_payload["uninterrupted_arm_digest"] != trace.digest:
                raise ReplayCanonicalError(
                    "canonical trace digest differs from replay arm"
                )
        encoded = data.encode("utf-8") if isinstance(data, str) else data
        if encoded != canonical_json_bytes(payload):
            raise ReplayCanonicalError("replay receipt is not canonically serialized")
        return payload


__all__ = [
    "ArchitectureCarryIdentityEvidence",
    "CrossModeComparisonEvidence",
    "ExperimentIdentityEvidence",
    "HFPreservationEvidence",
    "OptimizerConfigEvidence",
    "REPLAY_SCHEMA_VERSION",
    "ReplayArmEvidence",
    "ReplayBlocker",
    "ReplayModeEvidence",
    "ReplayRunEvidence",
    "ReplayStepEvidence",
    "ReplayVerificationResult",
    "RngEvidence",
    "RuntimeEvidence",
    "StatefulReplayProof",
    "StatefulReplayReceipt",
    "ToleranceEvidence",
    "VerifierEvidence",
]
