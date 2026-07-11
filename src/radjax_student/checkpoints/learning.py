"""Deterministic layered checkpoint persistence for P3.6."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.architecture import ArchitectureState
from radjax_student.learning import LearningState
from radjax_student.optimizers import OptimizerState

CHECKPOINT_SCHEMA_VERSION = "learning_checkpoint.v1"
CHECKPOINT_FILES = (
    "architecture.json",
    "learning.json",
    "optimizer.json",
    "manifest.json",
)


@dataclass(frozen=True)
class LearningCheckpoint:
    runtime_reference: str
    learning_state: LearningState
    architecture_state: ArchitectureState | None
    optimizer_state: OptimizerState
    parameters: Mapping[str, float]
    manifest: Mapping[str, Any]
    integrity: Mapping[str, str]
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not self.runtime_reference
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "checkpoint runtime reference or schema version is invalid"
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
            },
            "optimizer.json": {
                "optimizer_state": self.optimizer_state.to_dict(),
                "backend_state": self.optimizer_state.backend_state,
            },
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
        "files": list(payloads),
        "hashes": hashes,
        "sizes": sizes,
        "ownership": {
            "architecture.json": "architecture",
            "learning.json": "learning",
            "optimizer.json": "optimizer",
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
        manifest,
        integrity,
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
    for name, expected in manifest_payload["hashes"].items():
        data = (directory / name).read_bytes()
        if _digest(data) != expected:
            raise ValueError("checkpoint component hash mismatch")
    architecture, learning, optimizer = (
        _read(directory / name)
        for name in ("architecture.json", "learning.json", "optimizer.json")
    )
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
        manifest_payload,
        integrity,
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
    return json.loads(path.read_text())
