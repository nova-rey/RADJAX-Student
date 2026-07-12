"""Deterministic layered checkpoint persistence for P3.6."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints.roles import (
    CONTINUATION_CHECKPOINT_ROLE,
    HF_DISTRIBUTION_CHECKPOINT_ROLE,
    CheckpointPayloadDescriptor,
)
from radjax_student.learning import LearningState
from radjax_student.optimizers import OptimizerState

CHECKPOINT_SCHEMA_VERSION = "learning_checkpoint.v2"
CHECKPOINT_FILES = (
    "architecture.json",
    "learning.json",
    "optimizer.json",
    "source.json",
    "manifest.json",
)
_COMPONENT_FILES = CHECKPOINT_FILES[:-1]
_OWNERSHIP = {
    "architecture.json": "architecture",
    "learning.json": "learning",
    "optimizer.json": "optimizer",
    "source.json": "batch_source",
}
_PAYLOAD_DESCRIPTORS = {
    "architecture.json": CheckpointPayloadDescriptor(
        "architecture", "json", "scalar_parameter_mapping"
    ),
    "learning.json": CheckpointPayloadDescriptor("learning", "json", "state"),
    "optimizer.json": CheckpointPayloadDescriptor("optimizer", "json", "state"),
    "source.json": CheckpointPayloadDescriptor("batch_source", "json", "source_state"),
}


@dataclass(frozen=True)
class LearningCheckpoint:
    runtime_reference: str
    learning_state: LearningState
    architecture_state: ArchitectureState | None
    optimizer_state: OptimizerState
    parameters: Mapping[str, float]
    source_state: Mapping[str, Any] | None
    manifest: Mapping[str, Any]
    integrity: Mapping[str, str]
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    role: str = CONTINUATION_CHECKPOINT_ROLE

    def __post_init__(self) -> None:
        if (
            not self.runtime_reference
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "checkpoint runtime reference or schema version is invalid"
            )
        if self.role not in {
            CONTINUATION_CHECKPOINT_ROLE,
            HF_DISTRIBUTION_CHECKPOINT_ROLE,
        }:
            raise ValueError("unsupported checkpoint role")
        if self.role == HF_DISTRIBUTION_CHECKPOINT_ROLE:
            raise ValueError(
                "HF distribution checkpoints require an explicit conversion "
                "boundary and cannot use the continuation payload"
            )
        if self.learning_state.optimizer_step != self.optimizer_state.step:
            raise ValueError("learning and optimizer step continuity mismatch")
        if set(self.parameters) != set(self.optimizer_state.parameter_paths):
            raise ValueError("architecture parameters and optimizer paths must match")
        if not all(
            isinstance(path, str) and isinstance(value, (int, float))
            for path, value in self.parameters.items()
        ):
            raise TypeError("checkpoint parameters must be scalar path mappings")
        object.__setattr__(
            self, "source_state", _freeze_source_state(self.source_state)
        )

    def payloads(self) -> dict[str, dict[str, Any]]:
        return {
            "architecture.json": {
                "architecture_state": None
                if self.architecture_state is None
                else self.architecture_state.to_dict(),
                "parameters": dict(sorted(self.parameters.items())),
            },
            "learning.json": {
                "runtime_reference": self.runtime_reference,
                "learning_state": self.learning_state.to_dict(),
                "checkpoint_role": self.role,
            },
            "optimizer.json": {
                "optimizer_state": self.optimizer_state.to_dict(),
                "backend_state": self.optimizer_state.backend_state,
            },
            "source.json": {"source_state": _json_value(self.source_state)},
        }


def save_learning_checkpoint(
    checkpoint: LearningCheckpoint, directory: Path
) -> LearningCheckpoint:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = checkpoint.payloads()
    hashes = {name: _digest(_encode(payload)) for name, payload in payloads.items()}
    sizes = {name: len(_encode(payload)) for name, payload in payloads.items()}
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": checkpoint.role,
        "files": list(payloads),
        "hashes": hashes,
        "sizes": sizes,
        "ownership": _OWNERSHIP,
        "payload_descriptors": {
            name: descriptor.to_dict()
            for name, descriptor in _PAYLOAD_DESCRIPTORS.items()
        },
    }
    integrity = {"algorithm": "sha256", "manifest_digest": _digest(_encode(manifest))}
    for name, payload in payloads.items():
        (directory / name).write_bytes(_encode(payload))
    (directory / "manifest.json").write_bytes(
        _encode({**manifest, "integrity": integrity})
    )
    return LearningCheckpoint(
        checkpoint.runtime_reference,
        checkpoint.learning_state,
        checkpoint.architecture_state,
        checkpoint.optimizer_state,
        checkpoint.parameters,
        checkpoint.source_state,
        manifest,
        integrity,
        role=checkpoint.role,
    )


def load_learning_checkpoint(
    directory: Path, *, runtime_reference: str | None = None
) -> LearningCheckpoint:
    manifest_payload = _read(directory / "manifest.json")
    integrity = manifest_payload.pop("integrity")
    if integrity.get("manifest_digest") != _digest(_encode(manifest_payload)):
        raise ValueError("checkpoint manifest hash mismatch")
    if manifest_payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    _validate_manifest(manifest_payload)
    for name, expected in manifest_payload["hashes"].items():
        if not (directory / name).is_file():
            raise ValueError("checkpoint component missing")
        data = (directory / name).read_bytes()
        if _digest(data) != expected:
            raise ValueError("checkpoint component hash mismatch")
        if len(data) != manifest_payload["sizes"][name]:
            raise ValueError("checkpoint component size mismatch")
    architecture, learning, optimizer, source = (
        _read(directory / name) for name in _COMPONENT_FILES
    )
    if not isinstance(source, Mapping) or set(source) != {"source_state"}:
        raise ValueError("checkpoint source component is invalid")
    if (
        runtime_reference is not None
        and learning["runtime_reference"] != runtime_reference
    ):
        raise ValueError("checkpoint runtime reference mismatch")
    architecture_state = architecture["architecture_state"]
    state = OptimizerState(
        **_optimizer_kwargs(optimizer["optimizer_state"], optimizer["backend_state"])
    )
    return LearningCheckpoint(
        learning["runtime_reference"],
        LearningState.from_dict(learning["learning_state"]),
        None
        if architecture_state is None
        else ArchitectureState.from_dict(architecture_state),
        state,
        architecture["parameters"],
        source["source_state"],
        manifest_payload,
        integrity,
        role=manifest_payload.get("checkpoint_role", CONTINUATION_CHECKPOINT_ROLE),
    )


def _optimizer_kwargs(payload: Mapping[str, Any], backend_state: Any) -> dict[str, Any]:
    return {
        "optimizer_id": payload["optimizer_id"],
        "parameter_paths": tuple(payload["parameter_paths"]),
        "step": payload["step"],
        "schema_version": payload["schema_version"],
        "state_structure": payload["state_structure"],
        "backend_state": backend_state,
        "metadata": payload["metadata"],
        "claims_not_made": tuple(payload["claims_not_made"]),
    }


def _encode(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint component is unreadable") from exc


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if tuple(manifest.get("files", ())) != _COMPONENT_FILES:
        raise ValueError("checkpoint manifest components are invalid")
    if set(manifest.get("hashes", ())) != set(_COMPONENT_FILES):
        raise ValueError("checkpoint manifest hashes are invalid")
    if set(manifest.get("sizes", ())) != set(_COMPONENT_FILES):
        raise ValueError("checkpoint manifest sizes are invalid")
    if manifest.get("ownership") != _OWNERSHIP:
        raise ValueError("checkpoint manifest ownership is invalid")
    descriptors = manifest.get("payload_descriptors")
    expected_descriptors = {
        name: descriptor.to_dict() for name, descriptor in _PAYLOAD_DESCRIPTORS.items()
    }
    if descriptors is not None and descriptors != expected_descriptors:
        raise ValueError("checkpoint payload descriptors are invalid")
    if manifest.get("checkpoint_role", CONTINUATION_CHECKPOINT_ROLE) != (
        CONTINUATION_CHECKPOINT_ROLE
    ):
        raise ValueError("checkpoint role is not a continuation role")


def _freeze_source_state(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint source state must be a mapping or None")
    return _freeze_mapping(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint source state keys must be strings")
        normalized[key] = _freeze_value(item)
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint source state values must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    raise TypeError("checkpoint source state must be JSON-safe")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
