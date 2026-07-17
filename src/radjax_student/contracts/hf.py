"""Dependency-neutral Hugging Face lifecycle identity contracts.

These contracts deliberately describe compatibility identity only.  They do
not import architecture implementations, checkpoint code, JAX, or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

HF_DESCRIPTOR_SCHEMA_VERSION = "hf_compatibility_descriptor.v2"
HF_REFERENCE_SCHEMA_VERSION = "hf_preservation_reference.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECTION_RULES = frozenset({"identity", "transpose_2d", "reshape"})
_UNKNOWN = "identity_unavailable"


class HFContractError(ValueError):
    """Stable neutral HF identity failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


# Historical public spelling remains an exact alias, not a second hierarchy.
HFCompatibilityError = HFContractError


def canonical_hf_json(value: Any) -> bytes:
    """Encode finite, JSON-safe contract values deterministically."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def hf_digest(value: Any) -> str:
    return hashlib.sha256(canonical_hf_json(value)).hexdigest()


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFContractError("hf_descriptor_invalid", f"{name} must be nonempty")
    return value


def _digest(value: object, name: str) -> str:
    value = _string(value, name)
    if not _SHA256.fullmatch(value):
        raise HFContractError("hf_descriptor_invalid", f"{name} must be sha256")
    return value


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HFContractError("hf_descriptor_invalid", "metadata must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise HFContractError(
                "hf_descriptor_invalid", "metadata keys must be strings"
            )
        return MappingProxyType(
            {key: _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise HFContractError("hf_descriptor_invalid", "metadata must be JSON-safe")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strict(payload: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    actual = frozenset(payload)
    if actual != fields:
        missing, unknown = sorted(fields - actual), sorted(actual - fields)
        raise HFContractError(
            "hf_descriptor_invalid",
            f"{label} fields are invalid: missing={missing}, unknown={unknown}",
        )


def _path(value: object, name: str) -> str:
    value = _string(value, name)
    if value.startswith(("/", ".")) or "//" in value or ".." in value.split("/"):
        raise HFContractError("hf_descriptor_invalid", f"{name} must be stable")
    if any(
        part in {"mesh", "shard", "device", "process", "buffer_layout"}
        for part in re.split(r"[/.]", value)
    ):
        raise HFContractError(
            "hf_parameter_projection_mismatch",
            f"{name} must not encode runtime layout",
        )
    return value


@dataclass(frozen=True)
class HFTokenizerIdentity:
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_content_digest: str
    tokenizer_config_digest: str
    tokenizer_family: str
    normalization_digest: str
    identity_availability: str = "known"

    _FIELDS = frozenset(
        {
            "tokenizer_id",
            "tokenizer_revision",
            "tokenizer_content_digest",
            "tokenizer_config_digest",
            "tokenizer_family",
            "normalization_digest",
            "identity_availability",
        }
    )

    def __post_init__(self) -> None:
        for name in ("tokenizer_id", "tokenizer_revision", "tokenizer_family"):
            _string(getattr(self, name), name)
        for name in (
            "tokenizer_content_digest",
            "tokenizer_config_digest",
            "normalization_digest",
        ):
            value = getattr(self, name)
            if value != _UNKNOWN:
                _digest(value, name)
        if self.identity_availability not in {
            "known",
            "synthetic",
            "embedded",
            "identity_unavailable",
        }:
            raise HFContractError(
                "hf_tokenizer_identity_invalid", "identity_availability is unsupported"
            )
        unknown = any(
            getattr(self, name) == _UNKNOWN
            for name in (
                "tokenizer_content_digest",
                "tokenizer_config_digest",
                "normalization_digest",
            )
        )
        if unknown and self.identity_availability not in {
            "synthetic",
            "embedded",
            "identity_unavailable",
        }:
            raise HFContractError(
                "hf_tokenizer_identity_invalid",
                "unknown tokenizer identity requires an explicit contract",
            )

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFTokenizerIdentity:
        _strict(payload, cls._FIELDS, "tokenizer_identity")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFVocabularyIdentity:
    vocabulary_size: int
    vocabulary_content_digest: str
    token_to_id_digest: str
    added_token_digest: str
    reserved_token_range: str | None

    _FIELDS = frozenset(
        {
            "vocabulary_size",
            "vocabulary_content_digest",
            "token_to_id_digest",
            "added_token_digest",
            "reserved_token_range",
        }
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.vocabulary_size, int)
            or isinstance(self.vocabulary_size, bool)
            or self.vocabulary_size <= 0
        ):
            raise HFContractError(
                "hf_vocabulary_identity_invalid", "vocabulary_size must be positive"
            )
        for name in (
            "vocabulary_content_digest",
            "token_to_id_digest",
            "added_token_digest",
        ):
            _digest(getattr(self, name), name)
        if self.reserved_token_range is not None:
            _string(self.reserved_token_range, "reserved_token_range")

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFVocabularyIdentity:
        _strict(payload, cls._FIELDS, "vocabulary_identity")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFSpecialTokenIdentity:
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None
    mask_token_id: int | None
    additional_special_token_ids: tuple[int, ...] = ()

    _FIELDS = frozenset(
        {
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "mask_token_id",
            "additional_special_token_ids",
        }
    )

    def __post_init__(self) -> None:
        values = [
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
            self.mask_token_id,
            *tuple(self.additional_special_token_ids),
        ]
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
            for value in values
        ):
            raise HFContractError(
                "hf_special_token_identity_invalid",
                "special token IDs must be nonnegative integers or absent",
            )
        actual = tuple(self.additional_special_token_ids)
        if len(actual) != len(set(actual)):
            raise HFContractError(
                "hf_special_token_identity_invalid",
                "additional special token IDs must be unique",
            )
        present = [value for value in values if value is not None]
        if len(present) != len(set(present)):
            raise HFContractError(
                "hf_special_token_identity_invalid",
                "special token assignments conflict",
            )
        object.__setattr__(self, "additional_special_token_ids", actual)

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def validate_for_vocabulary(self, vocabulary: HFVocabularyIdentity) -> None:
        values = [
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
            self.mask_token_id,
            *self.additional_special_token_ids,
        ]
        if any(
            value is not None and value >= vocabulary.vocabulary_size
            for value in values
        ):
            raise HFContractError(
                "hf_special_token_identity_invalid",
                "special token ID is outside vocabulary",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "mask_token_id": self.mask_token_id,
            "additional_special_token_ids": list(self.additional_special_token_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFSpecialTokenIdentity:
        _strict(payload, cls._FIELDS, "special_token_identity")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFParameterProjection:
    logical_path: str
    jax_keypath: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    exportability: str
    hf_distribution_key: str | None
    projection_rule: str
    tied_parameter_group: str | None = None
    non_exportability_reason: str | None = None

    _FIELDS = frozenset(
        {
            "logical_path",
            "jax_keypath",
            "shape",
            "dtype",
            "exportability",
            "hf_distribution_key",
            "projection_rule",
            "tied_parameter_group",
            "non_exportability_reason",
        }
    )

    def __post_init__(self) -> None:
        _path(self.logical_path, "logical_path")
        path = tuple(self.jax_keypath)
        if not path or any(not isinstance(part, str) or not part for part in path):
            raise HFContractError(
                "hf_parameter_projection_mismatch", "jax_keypath is invalid"
            )
        _path("/".join(path), "jax_keypath")
        shape = tuple(self.shape)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in shape
        ):
            raise HFContractError(
                "hf_parameter_projection_mismatch", "shape is invalid"
            )
        _string(self.dtype, "dtype")
        if self.exportability not in {"exportable", "non_exportable"}:
            raise HFContractError(
                "hf_parameter_projection_mismatch", "exportability is invalid"
            )
        if self.exportability == "exportable":
            if (
                self.hf_distribution_key is None
                or self.non_exportability_reason is not None
            ):
                raise HFContractError(
                    "hf_parameter_projection_mismatch",
                    "exportable projection requires exactly one HF key",
                )
        else:
            if (
                self.hf_distribution_key is not None
                or not self.non_exportability_reason
            ):
                raise HFContractError(
                    "hf_parameter_projection_mismatch",
                    "non-exportable projection requires a stable reason and no HF key",
                )
        if self.hf_distribution_key is not None:
            _path(self.hf_distribution_key, "hf_distribution_key")
        if self.projection_rule not in _PROJECTION_RULES:
            raise HFContractError(
                "hf_parameter_projection_mismatch", "projection_rule is unsupported"
            )
        if self.tied_parameter_group == "":
            raise HFContractError(
                "hf_parameter_projection_mismatch",
                "tied_parameter_group must be nonempty",
            )
        object.__setattr__(self, "jax_keypath", path)
        object.__setattr__(self, "shape", shape)

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "jax_keypath": list(self.jax_keypath),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "exportability": self.exportability,
            "hf_distribution_key": self.hf_distribution_key,
            "projection_rule": self.projection_rule,
            "tied_parameter_group": self.tied_parameter_group,
            "non_exportability_reason": self.non_exportability_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFParameterProjection:
        _strict(payload, cls._FIELDS, "parameter_projection")
        data = dict(payload)
        data["jax_keypath"] = tuple(data["jax_keypath"])
        data["shape"] = tuple(data["shape"])
        return cls(**data)


@dataclass(frozen=True)
class HFArchitectureProjection:
    config_family: str
    architecture_family: str
    hidden_size: int
    layer_count: int
    vocabulary_size: int
    context_length: int
    static_config: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS = frozenset(
        {
            "config_family",
            "architecture_family",
            "hidden_size",
            "layer_count",
            "vocabulary_size",
            "context_length",
            "static_config",
        }
    )

    def __post_init__(self) -> None:
        for name in ("config_family", "architecture_family"):
            _string(getattr(self, name), name)
        for name in ("hidden_size", "layer_count", "vocabulary_size", "context_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise HFContractError(
                    "hf_descriptor_invalid", f"{name} must be positive"
                )
        object.__setattr__(self, "static_config", _freeze(self.static_config))

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_family": self.config_family,
            "architecture_family": self.architecture_family,
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "static_config": _thaw(self.static_config),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFArchitectureProjection:
        _strict(payload, cls._FIELDS, "architecture_projection")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFPreservationReference:
    """Serialized derived projection; parsing is not lifecycle authority."""

    descriptor_schema_version: str
    descriptor_digest: str
    architecture_id: str
    model_type: str
    architecture_config_digest: str
    parameter_layout_digest: str
    tokenizer_identity_digest: str
    vocabulary_identity_digest: str
    special_token_identity_digest: str

    _FIELDS = frozenset(
        {
            "descriptor_schema_version",
            "descriptor_digest",
            "architecture_id",
            "model_type",
            "architecture_config_digest",
            "parameter_layout_digest",
            "tokenizer_identity_digest",
            "vocabulary_identity_digest",
            "special_token_identity_digest",
        }
    )

    def __post_init__(self) -> None:
        if self.descriptor_schema_version != HF_DESCRIPTOR_SCHEMA_VERSION:
            raise HFContractError(
                "hf_reference_derivation_mismatch",
                "unsupported HF preservation reference schema",
            )
        for name in ("architecture_id", "model_type"):
            _string(getattr(self, name), name)
        for name in self._FIELDS - {
            "descriptor_schema_version",
            "architecture_id",
            "model_type",
        }:
            _digest(getattr(self, name), name)

    @property
    def digest(self) -> str:
        return hf_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFPreservationReference:
        _strict(payload, cls._FIELDS, "hf_preservation_reference")
        return cls(**dict(payload))


@dataclass(frozen=True)
class HFCompatibilityDescriptor:
    schema_version: str
    architecture_id: str
    architecture_plugin_version: int
    model_type: str
    architecture_config_digest: str
    parameter_catalog_digest: str
    parameter_layout_digest: str
    tokenizer: HFTokenizerIdentity
    vocabulary: HFVocabularyIdentity
    special_tokens: HFSpecialTokenIdentity
    parameter_projections: tuple[HFParameterProjection, ...]
    architecture_projection: HFArchitectureProjection
    non_claims: tuple[str, ...] = ()
    notes: str | None = field(default=None, compare=False)

    _FIELDS = frozenset(
        {
            "schema_version",
            "architecture_id",
            "architecture_plugin_version",
            "model_type",
            "architecture_config_digest",
            "parameter_catalog_digest",
            "parameter_layout_digest",
            "tokenizer",
            "vocabulary",
            "special_tokens",
            "parameter_projections",
            "architecture_projection",
            "non_claims",
            "notes",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != HF_DESCRIPTOR_SCHEMA_VERSION:
            raise HFContractError(
                "hf_descriptor_schema_mismatch", "unsupported HF descriptor schema"
            )
        for name in ("architecture_id", "model_type"):
            _string(getattr(self, name), name)
        if (
            not isinstance(self.architecture_plugin_version, int)
            or self.architecture_plugin_version <= 0
        ):
            raise HFContractError(
                "hf_architecture_identity_mismatch",
                "architecture plugin version is invalid",
            )
        for name in (
            "architecture_config_digest",
            "parameter_catalog_digest",
            "parameter_layout_digest",
        ):
            _digest(getattr(self, name), name)
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (self.tokenizer, HFTokenizerIdentity),
                (self.vocabulary, HFVocabularyIdentity),
                (self.special_tokens, HFSpecialTokenIdentity),
                (self.architecture_projection, HFArchitectureProjection),
            )
        ):
            raise HFContractError(
                "hf_descriptor_invalid", "descriptor subcontracts are invalid"
            )
        self.special_tokens.validate_for_vocabulary(self.vocabulary)
        if (
            self.architecture_projection.vocabulary_size
            != self.vocabulary.vocabulary_size
        ):
            raise HFContractError(
                "hf_vocabulary_identity_mismatch",
                "architecture projection vocabulary differs",
            )
        projections = tuple(
            sorted(self.parameter_projections, key=lambda item: item.logical_path)
        )
        if not projections or any(
            not isinstance(item, HFParameterProjection) for item in projections
        ):
            raise HFContractError(
                "hf_parameter_projection_mismatch",
                "parameter projections must be nonempty",
            )
        if len({item.logical_path for item in projections}) != len(projections) or len(
            {item.jax_keypath for item in projections}
        ) != len(projections):
            raise HFContractError(
                "hf_parameter_projection_mismatch", "parameter paths must be unique"
            )
        keys = [
            item.hf_distribution_key
            for item in projections
            if item.hf_distribution_key is not None
        ]
        duplicate_keys = {key for key in keys if keys.count(key) > 1}
        for key in duplicate_keys:
            group = {
                item.tied_parameter_group
                for item in projections
                if item.hf_distribution_key == key
            }
            if len(group) != 1 or None in group:
                raise HFContractError(
                    "hf_parameter_projection_mismatch",
                    "shared HF key requires one explicit tied group",
                )
        claims = tuple(sorted(set(self.non_claims)))
        if any(not isinstance(value, str) or not value for value in claims):
            raise HFContractError(
                "hf_descriptor_invalid", "non_claims must contain stable identifiers"
            )
        if self.notes is not None:
            _string(self.notes, "notes")
        object.__setattr__(self, "parameter_projections", projections)
        object.__setattr__(self, "non_claims", claims)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture_id": self.architecture_id,
            "architecture_plugin_version": self.architecture_plugin_version,
            "model_type": self.model_type,
            "architecture_config_digest": self.architecture_config_digest,
            "parameter_catalog_digest": self.parameter_catalog_digest,
            "parameter_layout_digest": self.parameter_layout_digest,
            "tokenizer": self.tokenizer.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "special_tokens": self.special_tokens.to_dict(),
            "parameter_projections": [
                item.to_dict() for item in self.parameter_projections
            ],
            "architecture_projection": self.architecture_projection.to_dict(),
            "non_claims": list(self.non_claims),
        }

    @property
    def digest(self) -> str:
        return hf_digest(self.identity_payload())

    @property
    def parameter_projection_digest(self) -> str:
        return hf_digest([item.to_dict() for item in self.parameter_projections])

    def preservation_reference(self) -> HFPreservationReference:
        return HFPreservationReference(
            HF_DESCRIPTOR_SCHEMA_VERSION,
            self.digest,
            self.architecture_id,
            self.model_type,
            self.architecture_config_digest,
            self.parameter_layout_digest,
            self.tokenizer.digest,
            self.vocabulary.digest,
            self.special_tokens.digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "notes": self.notes}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HFCompatibilityDescriptor:
        _strict(payload, cls._FIELDS, "hf_compatibility_descriptor")
        data = dict(payload)
        data["tokenizer"] = HFTokenizerIdentity.from_dict(data["tokenizer"])
        data["vocabulary"] = HFVocabularyIdentity.from_dict(data["vocabulary"])
        data["special_tokens"] = HFSpecialTokenIdentity.from_dict(
            data["special_tokens"]
        )
        data["parameter_projections"] = tuple(
            HFParameterProjection.from_dict(item)
            for item in data["parameter_projections"]
        )
        data["architecture_projection"] = HFArchitectureProjection.from_dict(
            data["architecture_projection"]
        )
        return cls(**data)

    def to_json(self) -> str:
        return canonical_hf_json(self.to_dict()).decode().rstrip("\n")

    @classmethod
    def from_json(cls, value: str) -> HFCompatibilityDescriptor:
        return cls.from_dict(json.loads(value))


# P3.5 spelling retained as an alias to the canonical projection type.
HFParameterMapping = HFParameterProjection


__all__ = [
    "HFCompatibilityDescriptor",
    "HFCompatibilityError",
    "HFContractError",
    "HF_DESCRIPTOR_SCHEMA_VERSION",
    "HF_REFERENCE_SCHEMA_VERSION",
    "HFArchitectureProjection",
    "HFParameterMapping",
    "HFParameterProjection",
    "HFPreservationReference",
    "HFSpecialTokenIdentity",
    "HFTokenizerIdentity",
    "HFVocabularyIdentity",
    "canonical_hf_json",
    "hf_digest",
]
