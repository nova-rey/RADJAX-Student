"""Dependency-free Hugging Face preservation descriptors.

This module describes a logical export contract. It does not import or invoke
Transformers, safetensors, a runtime backend, or an architecture plugin.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import ArchitectureConfig, ParameterCatalog

HF_DESCRIPTOR_SCHEMA_VERSION = "hf_compatibility_descriptor.v1"
_RUNTIME_NAME_TOKENS = {
    "buffer",
    "device",
    "fused",
    "host",
    "kernel",
    "mesh",
    "partition",
    "process",
    "process_index",
    "replica",
    "shard",
}


class HFCompatibilityError(ValueError):
    """Raised when an HF projection cannot preserve logical identity."""


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HFCompatibilityError("metadata must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise HFCompatibilityError("metadata keys must be strings")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(dict(sorted(frozen.items())))


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HFCompatibilityError("metadata values must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    raise HFCompatibilityError("metadata must be finite JSON-safe values")


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _validate_logical_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not name:
        raise HFCompatibilityError(f"{label} must be a non-empty string")
    tokens = {
        token
        for segment in re.split(r"[./:_-]+", name.lower())
        for token in segment.split("_")
        if token
    }
    if tokens & _RUNTIME_NAME_TOKENS:
        raise HFCompatibilityError(f"{label} must be runtime-layout independent")


@dataclass(frozen=True)
class HFParameterMapping:
    logical_path: str
    jax_pytree_path: str
    hf_distribution_key: str
    shape: tuple[int, ...]
    dtype: str
    tied_weight_group: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _validate_logical_name(self.logical_path, "logical_path")
        _validate_logical_name(self.jax_pytree_path, "jax_pytree_path")
        _validate_logical_name(self.hf_distribution_key, "hf_distribution_key")
        if not self.shape or any(
            not isinstance(item, int) or item <= 0 for item in self.shape
        ):
            raise HFCompatibilityError("shape dimensions must be positive integers")
        if not isinstance(self.tied_weight_group, (str, type(None))):
            raise HFCompatibilityError("tied_weight_group must be a string or None")
        if self.tied_weight_group == "":
            raise HFCompatibilityError("tied_weight_group must be nonempty when set")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise HFCompatibilityError("dtype must be a non-empty string")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "jax_pytree_path": self.jax_pytree_path,
            "hf_distribution_key": self.hf_distribution_key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tied_weight_group": self.tied_weight_group,
            "metadata": _json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFParameterMapping:
        return cls(
            logical_path=str(payload["logical_path"]),
            jax_pytree_path=str(payload["jax_pytree_path"]),
            hf_distribution_key=str(payload["hf_distribution_key"]),
            shape=tuple(payload["shape"]),
            dtype=str(payload["dtype"]),
            tied_weight_group=payload.get("tied_weight_group"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class HFCompatibilityDescriptor:
    model_type: str
    architecture_id: str
    tokenizer_id: str
    vocab_size: int
    special_token_ids: Mapping[str, int]
    parameter_mappings: tuple[HFParameterMapping, ...]
    architecture_projection: Mapping[str, Any]
    architecture_state_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    compatibility_version: str = "1"
    schema_version: str = HF_DESCRIPTOR_SCHEMA_VERSION
    unknown_fields: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.model_type, "model_type"),
            (self.architecture_id, "architecture_id"),
            (self.tokenizer_id, "tokenizer_id"),
        ):
            _validate_logical_name(value, label)
        if self.schema_version != HF_DESCRIPTOR_SCHEMA_VERSION:
            raise HFCompatibilityError("unsupported HF descriptor schema")
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise HFCompatibilityError("vocab_size must be positive")
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self.vocab_size
            for key, value in self.special_token_ids.items()
        ):
            raise HFCompatibilityError("special token IDs must be in the vocabulary")
        if len(set(self.special_token_ids.values())) != len(self.special_token_ids):
            raise HFCompatibilityError("duplicate special token IDs require aliases")
        mappings = tuple(self.parameter_mappings)
        if not mappings or any(
            not isinstance(item, HFParameterMapping) for item in mappings
        ):
            raise HFCompatibilityError("parameter mappings must be nonempty mappings")
        for field_name in ("logical_path", "jax_pytree_path", "hf_distribution_key"):
            values = [getattr(item, field_name) for item in mappings]
            if len(values) != len(set(values)):
                raise HFCompatibilityError(f"{field_name} values must be unique")
        object.__setattr__(self, "special_token_ids", _freeze(self.special_token_ids))
        object.__setattr__(self, "parameter_mappings", mappings)
        object.__setattr__(
            self, "architecture_projection", _freeze(self.architecture_projection)
        )
        object.__setattr__(
            self,
            "architecture_state_metadata",
            _freeze(self.architecture_state_metadata),
        )
        object.__setattr__(self, "unknown_fields", _freeze(self.unknown_fields))

    @classmethod
    def from_architecture(
        cls,
        config: ArchitectureConfig,
        parameter_catalog: ParameterCatalog,
        *,
        model_type: str,
        tokenizer_id: str,
        special_token_ids: Mapping[str, int],
        parameter_mappings: tuple[HFParameterMapping, ...],
        architecture_state_metadata: Mapping[str, Any] | None = None,
    ) -> HFCompatibilityDescriptor:
        projection = {
            "architecture_id": config.architecture_id,
            "model_config": config.model_config,
            "vocab_size": config.vocab_size,
            "sequence_length": config.sequence_length,
            "dtype_intent": config.dtype_intent,
        }
        if config.vocab_size is None:
            raise HFCompatibilityError("architecture config must declare vocab_size")
        descriptor = cls(
            model_type=model_type,
            architecture_id=config.architecture_id,
            tokenizer_id=tokenizer_id,
            vocab_size=config.vocab_size,
            special_token_ids=special_token_ids,
            parameter_mappings=parameter_mappings,
            architecture_projection=projection,
            architecture_state_metadata=architecture_state_metadata or {},
        )
        descriptor.validate_against(config, parameter_catalog)
        return descriptor

    def validate_against(
        self, config: ArchitectureConfig, parameter_catalog: ParameterCatalog
    ) -> None:
        expected = {
            "architecture_id": config.architecture_id,
            "model_config": _json(config.model_config),
            "vocab_size": config.vocab_size,
            "sequence_length": config.sequence_length,
            "dtype_intent": config.dtype_intent,
        }
        actual = _json(self.architecture_projection)
        if self.architecture_id != config.architecture_id or actual != expected:
            raise HFCompatibilityError(
                "HF descriptor conflicts with architecture configuration"
            )
        if parameter_catalog.architecture_id != config.architecture_id:
            raise HFCompatibilityError("parameter catalog conflicts with architecture")
        mappings = {item.logical_path: item for item in self.parameter_mappings}
        if set(mappings) != set(parameter_catalog.paths):
            raise HFCompatibilityError(
                "HF mappings must exactly cover parameter catalog"
            )
        for descriptor in parameter_catalog.parameters:
            mapping = mappings[descriptor.path]
            if mapping.shape != descriptor.shape:
                raise HFCompatibilityError("HF mapping shape conflicts with catalog")
            if mapping.dtype != descriptor.dtype:
                raise HFCompatibilityError("HF mapping dtype conflicts with catalog")

    def to_dict(self) -> dict[str, Any]:
        known = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "architecture_id": self.architecture_id,
            "tokenizer_id": self.tokenizer_id,
            "vocab_size": self.vocab_size,
            "special_token_ids": _json(self.special_token_ids),
            "parameter_mappings": [item.to_dict() for item in self.parameter_mappings],
            "architecture_projection": _json(self.architecture_projection),
            "architecture_state_metadata": _json(self.architecture_state_metadata),
            "compatibility_version": self.compatibility_version,
        }
        return {**_json(self.unknown_fields), **known}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFCompatibilityDescriptor:
        known = {
            "schema_version",
            "model_type",
            "architecture_id",
            "tokenizer_id",
            "vocab_size",
            "special_token_ids",
            "parameter_mappings",
            "architecture_projection",
            "architecture_state_metadata",
            "compatibility_version",
        }
        return cls(
            schema_version=str(
                payload.get("schema_version", HF_DESCRIPTOR_SCHEMA_VERSION)
            ),
            model_type=str(payload["model_type"]),
            architecture_id=str(payload["architecture_id"]),
            tokenizer_id=str(payload["tokenizer_id"]),
            vocab_size=payload["vocab_size"],
            special_token_ids=payload.get("special_token_ids", {}),
            parameter_mappings=tuple(
                HFParameterMapping.from_dict(item)
                for item in payload["parameter_mappings"]
            ),
            architecture_projection=payload.get("architecture_projection", {}),
            architecture_state_metadata=payload.get("architecture_state_metadata", {}),
            compatibility_version=str(payload.get("compatibility_version", "1")),
            unknown_fields={
                key: value for key, value in payload.items() if key not in known
            },
        )

    @classmethod
    def from_json(cls, value: str) -> HFCompatibilityDescriptor:
        return cls.from_dict(json.loads(value))


__all__ = [
    "HFCompatibilityDescriptor",
    "HFCompatibilityError",
    "HFParameterMapping",
]
