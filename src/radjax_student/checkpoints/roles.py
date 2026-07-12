"""Explicit checkpoint roles and versioned payload descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NoReturn

CONTINUATION_CHECKPOINT_ROLE = "radjax_continuation"
HF_DISTRIBUTION_CHECKPOINT_ROLE = "hf_distribution"
PAYLOAD_DESCRIPTOR_SCHEMA_VERSION = "checkpoint_payload_descriptor.v1"


@dataclass(frozen=True)
class CheckpointPayloadDescriptor:
    """Truthful description of a serialized continuation component."""

    owner: str
    codec: str
    kind: str
    schema_version: str = PAYLOAD_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAYLOAD_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint payload descriptor schema")
        if not all(
            isinstance(value, str) and value
            for value in (self.owner, self.codec, self.kind)
        ):
            raise ValueError("checkpoint payload descriptor fields must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "owner": self.owner,
            "codec": self.codec,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointPayloadDescriptor:
        return cls(
            schema_version=str(
                payload.get("schema_version", PAYLOAD_DESCRIPTOR_SCHEMA_VERSION)
            ),
            owner=str(payload["owner"]),
            codec=str(payload["codec"]),
            kind=str(payload["kind"]),
        )


@dataclass(frozen=True)
class FutureTensorPayloadDescriptor:
    """Reserved descriptor only; no tensor-pytree codec exists in P3.5."""

    logical_tree_schema: str
    storage_codec: str
    schema_version: str = "future_tensor_payload_descriptor.v1"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema_version != "future_tensor_payload_descriptor.v1":
            raise ValueError("unsupported future tensor payload descriptor schema")
        if not self.logical_tree_schema or not self.storage_codec:
            raise ValueError("future tensor descriptor fields must be nonempty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_tree_schema": self.logical_tree_schema,
            "storage_codec": self.storage_codec,
            "metadata": dict(self.metadata),
        }


def reject_implicit_hf_conversion(checkpoint: Any) -> NoReturn:
    """Refuse to use a continuation checkpoint as an HF distribution file."""

    if getattr(checkpoint, "role", None) != CONTINUATION_CHECKPOINT_ROLE:
        raise ValueError("checkpoint is not a RADJAX continuation checkpoint")
    raise ValueError(
        "explicit HF distribution conversion is not implemented; continuation "
        f"role cannot be treated as {HF_DISTRIBUTION_CHECKPOINT_ROLE}"
    )


__all__ = [
    "CONTINUATION_CHECKPOINT_ROLE",
    "CheckpointPayloadDescriptor",
    "FutureTensorPayloadDescriptor",
    "HF_DISTRIBUTION_CHECKPOINT_ROLE",
    "PAYLOAD_DESCRIPTOR_SCHEMA_VERSION",
    "reject_implicit_hf_conversion",
]
