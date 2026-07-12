"""Learning checkpoint v3 with typed optimizer-array sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radjax_student.architecture.models import ArchitectureState
from radjax_student.checkpoints.npz_codec import (
    descriptor_digest,
    read_deterministic_npz,
    write_deterministic_npz,
)
from radjax_student.contracts import (
    HFPreservationReference,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.learning.models import LearningState
from radjax_student.optimizers.jax import (
    JaxOptimizerState,
    validate_jax_optimizer_state,
)
from radjax_student.optimizers.models import OptimizerState
from radjax_student.optimizers.protocols import JaxOptimizerBackend

CHECKPOINT_V3_SCHEMA_VERSION = "learning_checkpoint.v3"
CHECKPOINT_OPTIMIZER_STEP_MISMATCH = "checkpoint_optimizer_step_mismatch"
ARCHITECTURE_CARRY_SCHEMA_VERSION = "architecture_carry.v1"
V3_FILES = (
    "parameters.npz",
    "parameters.json",
    "architecture_carry.npz",
    "architecture_carry.json",
    "optimizer_state.npz",
    "optimizer_state.json",
    "learning.json",
    "layout.json",
    "manifest.json",
)
V3_OWNERSHIP = {
    "parameters.npz": "architecture",
    "parameters.json": "architecture",
    "architecture_carry.npz": "architecture",
    "architecture_carry.json": "architecture",
    "optimizer_state.npz": "optimizer",
    "optimizer_state.json": "optimizer",
    "learning.json": "learning",
    "layout.json": "architecture",
}


class CheckpointValidationError(ValueError):
    """Structured v3 validation failure with a stable blocker code."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.details = MappingProxyType(dict(details or {}))
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class JaxLearningCheckpointV3:
    runtime_reference: str
    learning_state: LearningState
    optimizer_state: JaxOptimizerState
    parameters: Mapping[str, Any]
    architecture_carry: Mapping[str, Any]
    parameter_layout: ParameterTreeLayout
    architecture_state: ArchitectureState | None
    hf_reference: HFPreservationReference
    architecture_config_digest: str
    parameter_catalog_digest: str
    architecture_carry_descriptor: Mapping[str, Any] | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    integrity: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CHECKPOINT_V3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_reference:
            raise ValueError("runtime_reference must be nonempty")
        if self.schema_version != CHECKPOINT_V3_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint v3 schema")
        if not isinstance(self.learning_state, LearningState):
            raise TypeError("learning_state must be LearningState")
        if not isinstance(self.optimizer_state, JaxOptimizerState):
            raise TypeError("optimizer_state must be JaxOptimizerState")
        if not isinstance(self.parameter_layout, ParameterTreeLayout):
            raise TypeError("parameter_layout must be ParameterTreeLayout")
        if self.architecture_state is not None and not isinstance(
            self.architecture_state, ArchitectureState
        ):
            raise TypeError("architecture_state must be ArchitectureState when set")
        if not isinstance(self.hf_reference, HFPreservationReference):
            raise TypeError("hf_reference must be HFPreservationReference")
        for name in ("architecture_config_digest", "parameter_catalog_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.hf_reference.architecture_id != self.parameter_layout.architecture_id:
            raise ValueError("HF reference architecture identity does not match layout")
        if self.hf_reference.parameter_layout_digest != self.parameter_layout.digest():
            raise ValueError(
                "HF reference parameter layout digest does not match layout"
            )
        if (
            self.hf_reference.architecture_config_digest
            != self.architecture_config_digest
        ):
            raise ValueError(
                "HF reference architecture config digest does not match checkpoint"
            )
        if not isinstance(self.parameters, Mapping) or not isinstance(
            self.architecture_carry, Mapping
        ):
            raise TypeError("v3 tensor payloads must be mapping pytrees")
        if self.architecture_carry_descriptor is not None and not isinstance(
            self.architecture_carry_descriptor, Mapping
        ):
            raise TypeError("architecture_carry_descriptor must be a mapping")
        if self.architecture_carry_descriptor is not None:
            object.__setattr__(
                self,
                "architecture_carry_descriptor",
                MappingProxyType(dict(self.architecture_carry_descriptor)),
            )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "integrity", MappingProxyType(dict(self.integrity)))


def save_learning_checkpoint_v3(
    checkpoint: JaxLearningCheckpointV3,
    directory: Path,
    *,
    optimizer: JaxOptimizerBackend,
) -> JaxLearningCheckpointV3:
    """Validate then write one canonical v3 continuation checkpoint."""

    _validate_runtime_state(checkpoint, optimizer)
    if os.path.lexists(directory) and (
        not directory.is_dir() or any(directory.iterdir())
    ):
        raise CheckpointValidationError(
            "checkpoint_destination_exists",
            "refusing to mutate an existing checkpoint destination",
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent)
    )
    try:
        parameter_descriptor = write_deterministic_npz(
            temporary / "parameters.npz", checkpoint.parameters
        )
        carry_descriptor = write_deterministic_npz(
            temporary / "architecture_carry.npz", checkpoint.architecture_carry
        )
        optimizer_descriptor = write_deterministic_npz(
            temporary / "optimizer_state.npz", checkpoint.optimizer_state.arrays
        )
        architecture_carry_identity = (
            dict(checkpoint.architecture_carry_descriptor)
            if checkpoint.architecture_carry_descriptor is not None
            else {
                "schema_version": ARCHITECTURE_CARRY_SCHEMA_VERSION,
                "state_id": (
                    None
                    if checkpoint.architecture_state is None
                    else checkpoint.architecture_state.state_id
                ),
                "pytree_descriptor_digest": descriptor_digest(carry_descriptor),
            }
        )
        _validate_carry_identity(
            architecture_carry_identity,
            actual_descriptor=carry_descriptor,
            architecture_state=checkpoint.architecture_state,
        )
        descriptor = optimizer_state_descriptor_payload(
            checkpoint.optimizer_state,
            optimizer=optimizer,
            sidecar_digest=_digest((temporary / "optimizer_state.npz").read_bytes()),
            descriptor=optimizer_descriptor,
        )
        payloads = {
            "parameters.json": parameter_descriptor,
            "architecture_carry.json": carry_descriptor,
            "optimizer_state.json": descriptor,
            "learning.json": {
                "runtime_reference": checkpoint.runtime_reference,
                "learning_state": checkpoint.learning_state.to_dict(),
                "architecture_state": (
                    None
                    if checkpoint.architecture_state is None
                    else checkpoint.architecture_state.to_dict()
                ),
                "hf_reference": checkpoint.hf_reference.to_dict(),
                "architecture_config_digest": checkpoint.architecture_config_digest,
                "parameter_catalog_digest": checkpoint.parameter_catalog_digest,
                "architecture_carry_descriptor": architecture_carry_identity,
            },
            "layout.json": checkpoint.parameter_layout.to_dict(),
        }
        for name, payload in payloads.items():
            _write_json(temporary / name, payload)
        files = [name for name in V3_FILES if name != "manifest.json"]
        manifest = {
            "schema_version": CHECKPOINT_V3_SCHEMA_VERSION,
            "files": files,
            "ownership": V3_OWNERSHIP,
            "hashes": {
                name: _digest((temporary / name).read_bytes()) for name in files
            },
            "sizes": {name: (temporary / name).stat().st_size for name in files},
            "architecture": {
                "architecture_id": checkpoint.parameter_layout.architecture_id,
                "architecture_state_id": (
                    None
                    if checkpoint.architecture_state is None
                    else checkpoint.architecture_state.state_id
                ),
                "parameter_layout_digest": checkpoint.parameter_layout.digest(),
                "parameter_catalog_digest": checkpoint.parameter_catalog_digest,
                "architecture_config_digest": checkpoint.architecture_config_digest,
                "hf_reference": checkpoint.hf_reference.to_dict(),
                "parameters_descriptor_digest": descriptor_digest(parameter_descriptor),
                "carry_descriptor_digest": descriptor_digest(carry_descriptor),
                "architecture_carry_descriptor": architecture_carry_identity,
                "architecture_carry_identity_digest": descriptor_digest(
                    architecture_carry_identity
                ),
            },
            "optimizer": _optimizer_manifest(
                checkpoint.optimizer_state,
                optimizer=optimizer,
                sidecar_digest=_digest(
                    (temporary / "optimizer_state.npz").read_bytes()
                ),
                descriptor_digest=descriptor_digest(optimizer_descriptor),
            ),
        }
        integrity = {
            "algorithm": "sha256",
            "manifest_digest": _digest(_json_bytes(manifest)),
        }
        _write_json(temporary / "manifest.json", {**manifest, "integrity": integrity})
        load_learning_checkpoint_v3(
            temporary,
            optimizer=optimizer,
            parameter_layout=checkpoint.parameter_layout,
            runtime_reference=checkpoint.runtime_reference,
        )
        _fsync_tree(temporary)
        if directory.exists():
            directory.rmdir()
        os.rename(temporary, directory)
        _fsync_directory(directory.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return JaxLearningCheckpointV3(
        checkpoint.runtime_reference,
        checkpoint.learning_state,
        checkpoint.optimizer_state,
        checkpoint.parameters,
        checkpoint.architecture_carry,
        checkpoint.parameter_layout,
        checkpoint.architecture_state,
        checkpoint.hf_reference,
        checkpoint.architecture_config_digest,
        checkpoint.parameter_catalog_digest,
        architecture_carry_identity,
        manifest,
        integrity,
    )


def load_learning_checkpoint_v3(
    directory: Path,
    *,
    optimizer: JaxOptimizerBackend,
    parameter_layout: ParameterTreeLayout,
    runtime_reference: str | None = None,
    expected_hf_reference: HFPreservationReference | None = None,
    expected_architecture_config_digest: str | None = None,
    expected_parameter_catalog_digest: str | None = None,
    expected_architecture_state_id: str | None = None,
    expected_architecture_carry_descriptor: Mapping[str, Any] | None = None,
) -> JaxLearningCheckpointV3:
    """Validate all v3 identity, integrity, and optimizer-owned invariants."""

    stored = _read_json(directory / "manifest.json")
    integrity = stored.pop("integrity", None)
    if not isinstance(integrity, Mapping) or integrity.get(
        "manifest_digest"
    ) != _digest(_json_bytes(stored)):
        raise CheckpointValidationError(
            "checkpoint_manifest_hash_mismatch", "checkpoint manifest hash mismatch"
        )
    descriptor = optimizer.jax_state_descriptor(parameter_layout)
    _validate_manifest(
        stored,
        optimizer=optimizer,
        expected_descriptor=descriptor,
        expected_layout=parameter_layout,
    )
    for name in stored["files"]:
        path = directory / name
        if not path.is_file():
            raise CheckpointValidationError(
                "checkpoint_component_missing", "checkpoint component is missing"
            )
        data = path.read_bytes()
        if _digest(data) != stored["hashes"][name]:
            raise CheckpointValidationError(
                "checkpoint_component_hash_mismatch",
                "checkpoint component hash mismatch",
            )
        if len(data) != stored["sizes"][name]:
            raise CheckpointValidationError(
                "checkpoint_component_size_mismatch",
                "checkpoint component size mismatch",
            )
    stored_layout = _layout_from_dict(_read_json(directory / "layout.json"))
    if stored_layout.digest() != parameter_layout.digest():
        raise CheckpointValidationError(
            "checkpoint_layout_mismatch", "checkpoint parameter layout mismatch"
        )
    optimizer_payload = _read_json(directory / "optimizer_state.json")
    numerical_descriptor = optimizer_payload["numerical_state_descriptor"]
    if (
        optimizer_payload.get("optimizer_id") != optimizer.optimizer_id
        or optimizer_payload.get("optimizer_capability_version")
        != optimizer.optimizer_version
        or optimizer_payload.get("optimizer_numerical_state_schema_version")
        != descriptor.optimizer_schema_version
        or tuple(
            tuple(item["keypath"]) for item in numerical_descriptor.get("leaves", ())
        )
        != descriptor.state_keypaths
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_descriptor_mismatch",
            "optimizer numerical-state descriptor does not match the optimizer",
        )
    if optimizer_payload["descriptor_digest"] != descriptor_digest(
        numerical_descriptor
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_descriptor_hash_mismatch",
            "optimizer descriptor hash mismatch",
        )
    sidecar_digest = _digest((directory / "optimizer_state.npz").read_bytes())
    if optimizer_payload["sidecar_digest"] != sidecar_digest:
        raise CheckpointValidationError(
            "checkpoint_optimizer_sidecar_hash_mismatch",
            "optimizer numerical-state sidecar hash mismatch",
        )
    optimizer_manifest = stored["optimizer"]
    manifest_step = optimizer_manifest["envelope_step"]
    payload_step = optimizer_payload.get("envelope_step")
    envelope_step = optimizer_payload.get("envelope", {}).get("step")
    if not (manifest_step == payload_step == envelope_step):
        raise CheckpointValidationError(
            CHECKPOINT_OPTIMIZER_STEP_MISMATCH,
            "checkpoint optimizer step records disagree",
            details={
                "expected_step": manifest_step,
                "observed_step": payload_step,
            },
        )
    if (
        optimizer_manifest.get("optimizer_numerical_state_schema_version")
        != (descriptor.optimizer_schema_version)
        or optimizer_manifest.get("numerical_state_descriptor_digest")
        != (descriptor_digest(numerical_descriptor))
        or optimizer_manifest.get("numerical_state_sidecar_digest") != sidecar_digest
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_descriptor_mismatch",
            "checkpoint optimizer descriptor identity mismatch",
        )
    arrays = read_deterministic_npz(
        directory / "optimizer_state.npz", numerical_descriptor
    )
    envelope = _optimizer_envelope(optimizer_payload["envelope"])
    descriptor = optimizer.jax_state_descriptor(parameter_layout)
    state = JaxOptimizerState(envelope, descriptor, arrays)
    try:
        validate_jax_optimizer_state(
            state,
            optimizer=optimizer,
            optimizer_id=optimizer.optimizer_id,
            parameter_layout=parameter_layout,
            descriptor=descriptor,
        )
    except Exception as exc:
        _raise_optimizer_validation(exc)
    learning_payload = _read_json(directory / "learning.json")
    if (
        runtime_reference is not None
        and learning_payload["runtime_reference"] != runtime_reference
    ):
        raise CheckpointValidationError(
            "checkpoint_runtime_reference_mismatch",
            "checkpoint runtime reference mismatch",
        )
    learning_state = LearningState.from_dict(learning_payload["learning_state"])
    if learning_state.optimizer_step != envelope.step:
        raise CheckpointValidationError(
            CHECKPOINT_OPTIMIZER_STEP_MISMATCH,
            "learning and optimizer steps disagree",
            details={
                "expected_step": learning_state.optimizer_step,
                "observed_step": envelope.step,
            },
        )
    architecture_state_payload = learning_payload.get("architecture_state")
    architecture_state = (
        None
        if architecture_state_payload is None
        else ArchitectureState.from_dict(architecture_state_payload)
    )
    hf_reference = HFPreservationReference.from_dict(learning_payload["hf_reference"])
    architecture_config_digest = str(learning_payload["architecture_config_digest"])
    parameter_catalog_digest = str(learning_payload["parameter_catalog_digest"])
    architecture_carry_descriptor = learning_payload.get(
        "architecture_carry_descriptor"
    )
    if not isinstance(architecture_carry_descriptor, Mapping):
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_missing",
            "architecture carry descriptor is required",
        )
    parameters = read_deterministic_npz(
        directory / "parameters.npz", _read_json(directory / "parameters.json")
    )
    parameter_layout.validate_materialized_parameters(parameters)
    carry_payload = _read_json(directory / "architecture_carry.json")
    carry = read_deterministic_npz(directory / "architecture_carry.npz", carry_payload)
    architecture_manifest = stored["architecture"]
    if architecture_manifest["carry_descriptor_digest"] != descriptor_digest(
        carry_payload
    ):
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_mismatch",
            "architecture carry descriptor digest mismatch",
        )
    _validate_carry_identity(
        architecture_carry_descriptor,
        actual_descriptor=carry_payload,
        architecture_state=architecture_state,
    )
    _validate_lifecycle_identity(
        parameter_layout=parameter_layout,
        architecture_state=architecture_state,
        hf_reference=hf_reference,
        architecture_config_digest=architecture_config_digest,
        parameter_catalog_digest=parameter_catalog_digest,
        architecture_carry_descriptor=architecture_carry_descriptor,
        manifest=stored,
    )
    _validate_expected_lifecycle_identity(
        hf_reference=hf_reference,
        architecture_config_digest=architecture_config_digest,
        parameter_catalog_digest=parameter_catalog_digest,
        architecture_state=architecture_state,
        architecture_carry_descriptor=architecture_carry_descriptor,
        expected_hf_reference=expected_hf_reference,
        expected_architecture_config_digest=expected_architecture_config_digest,
        expected_parameter_catalog_digest=expected_parameter_catalog_digest,
        expected_architecture_state_id=expected_architecture_state_id,
        expected_architecture_carry_descriptor=expected_architecture_carry_descriptor,
    )
    return JaxLearningCheckpointV3(
        learning_payload["runtime_reference"],
        learning_state,
        state,
        parameters,
        carry,
        parameter_layout,
        architecture_state,
        hf_reference,
        architecture_config_digest,
        parameter_catalog_digest,
        architecture_carry_descriptor,
        stored,
        integrity,
    )


def optimizer_state_descriptor_payload(
    state: JaxOptimizerState,
    *,
    optimizer: JaxOptimizerBackend,
    sidecar_digest: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_V3_SCHEMA_VERSION,
        "optimizer_id": optimizer.optimizer_id,
        "optimizer_capability_version": optimizer.optimizer_version,
        "optimizer_numerical_state_schema_version": (
            state.descriptor.optimizer_schema_version
        ),
        "envelope": state.envelope.to_dict(),
        "envelope_step": state.envelope.step,
        "sidecar_digest": sidecar_digest,
        "descriptor_digest": descriptor_digest(descriptor),
        "numerical_state_descriptor": descriptor,
    }


def _optimizer_manifest(
    state: JaxOptimizerState,
    *,
    optimizer: JaxOptimizerBackend,
    sidecar_digest: str,
    descriptor_digest: str,
) -> dict[str, Any]:
    return {
        "optimizer_id": optimizer.optimizer_id,
        "optimizer_capability_version": optimizer.optimizer_version,
        "optimizer_numerical_state_schema_version": (
            state.descriptor.optimizer_schema_version
        ),
        "envelope_step": state.envelope.step,
        "numerical_state_sidecar_digest": sidecar_digest,
        "numerical_state_descriptor_digest": descriptor_digest,
    }


def _validate_runtime_state(
    checkpoint: JaxLearningCheckpointV3, optimizer: JaxOptimizerBackend
) -> None:
    descriptor = optimizer.jax_state_descriptor(checkpoint.parameter_layout)
    if checkpoint.optimizer_state.envelope.parameter_paths != (
        checkpoint.parameter_layout.logical_paths
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_parameter_paths_mismatch",
            "optimizer envelope parameter paths do not match layout",
        )
    if (
        checkpoint.learning_state.optimizer_step
        != checkpoint.optimizer_state.envelope.step
    ):
        raise CheckpointValidationError(
            CHECKPOINT_OPTIMIZER_STEP_MISMATCH,
            "learning and optimizer steps disagree",
            details={
                "expected_step": checkpoint.learning_state.optimizer_step,
                "observed_step": checkpoint.optimizer_state.envelope.step,
            },
        )
    checkpoint.parameter_layout.validate_materialized_parameters(checkpoint.parameters)
    try:
        validate_jax_optimizer_state(
            checkpoint.optimizer_state,
            optimizer=optimizer,
            optimizer_id=optimizer.optimizer_id,
            parameter_layout=checkpoint.parameter_layout,
            descriptor=descriptor,
        )
    except Exception as exc:
        _raise_optimizer_validation(exc)
    _validate_lifecycle_identity(
        parameter_layout=checkpoint.parameter_layout,
        architecture_state=checkpoint.architecture_state,
        hf_reference=checkpoint.hf_reference,
        architecture_config_digest=checkpoint.architecture_config_digest,
        parameter_catalog_digest=checkpoint.parameter_catalog_digest,
        architecture_carry_descriptor=checkpoint.architecture_carry_descriptor,
        manifest=None,
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    optimizer: JaxOptimizerBackend,
    expected_descriptor: Any,
    expected_layout: ParameterTreeLayout,
) -> None:
    if manifest.get("schema_version") != CHECKPOINT_V3_SCHEMA_VERSION:
        raise CheckpointValidationError(
            "checkpoint_schema_mismatch", "checkpoint schema mismatch"
        )
    if tuple(manifest.get("files", ())) != V3_FILES[:-1]:
        raise CheckpointValidationError(
            "checkpoint_manifest_invalid", "checkpoint files are invalid"
        )
    if manifest.get("ownership") != V3_OWNERSHIP:
        raise CheckpointValidationError(
            "checkpoint_manifest_invalid", "checkpoint ownership is invalid"
        )
    architecture_manifest = manifest.get("architecture", {})
    if architecture_manifest.get("parameter_layout_digest") != expected_layout.digest():
        raise CheckpointValidationError(
            "checkpoint_layout_mismatch", "checkpoint architecture layout mismatch"
        )
    optimizer_manifest = manifest.get("optimizer", {})
    if optimizer_manifest.get("optimizer_id") != optimizer.optimizer_id:
        raise CheckpointValidationError(
            "checkpoint_optimizer_identity_mismatch", "optimizer identity mismatch"
        )
    if (
        optimizer_manifest.get("optimizer_capability_version")
        != optimizer.optimizer_version
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_capability_mismatch",
            "optimizer capability version mismatch",
        )
    if optimizer_manifest.get("optimizer_numerical_state_schema_version") != (
        expected_descriptor.optimizer_schema_version
    ):
        raise CheckpointValidationError(
            "checkpoint_optimizer_schema_mismatch",
            "optimizer numerical-state schema mismatch",
        )
    if "envelope_step" not in optimizer_manifest:
        raise CheckpointValidationError(
            "checkpoint_manifest_invalid", "optimizer envelope step is missing"
        )


def _raise_optimizer_validation(exc: Exception) -> None:
    if getattr(exc, "code", None) == "optimizer_state_parameter_mismatch":
        raise CheckpointValidationError(
            "checkpoint_optimizer_parameter_paths_mismatch",
            "optimizer envelope parameter paths do not match layout",
        ) from exc
    if getattr(exc, "details", None) and "expected_step" in exc.details:
        raise CheckpointValidationError(
            CHECKPOINT_OPTIMIZER_STEP_MISMATCH,
            "optimizer envelope and numerical steps disagree",
            details={
                "expected_step": exc.details["expected_step"],
                "observed_step": exc.details["observed_step"],
            },
        ) from exc
    raise CheckpointValidationError(
        "checkpoint_optimizer_state_invalid", "optimizer-owned state validation failed"
    ) from exc


def _validate_lifecycle_identity(
    *,
    parameter_layout: ParameterTreeLayout,
    architecture_state: ArchitectureState | None,
    hf_reference: HFPreservationReference,
    architecture_config_digest: str,
    parameter_catalog_digest: str,
    architecture_carry_descriptor: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
) -> None:
    if hf_reference.architecture_id != parameter_layout.architecture_id:
        raise CheckpointValidationError(
            "checkpoint_hf_identity_mismatch",
            "HF architecture identity does not match checkpoint layout",
        )
    if hf_reference.parameter_layout_digest != parameter_layout.digest():
        raise CheckpointValidationError(
            "checkpoint_hf_layout_mismatch",
            "HF parameter layout identity does not match checkpoint layout",
        )
    if hf_reference.architecture_config_digest != architecture_config_digest:
        raise CheckpointValidationError(
            "checkpoint_hf_config_mismatch",
            "HF architecture config identity does not match checkpoint config",
        )
    if not parameter_catalog_digest:
        raise CheckpointValidationError(
            "checkpoint_catalog_identity_missing",
            "parameter catalog identity is required",
        )
    if not isinstance(architecture_carry_descriptor, Mapping) and manifest is not None:
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_missing",
            "architecture carry descriptor is required",
        )
    if (
        isinstance(architecture_carry_descriptor, Mapping)
        and architecture_state is not None
        and architecture_carry_descriptor.get("state_id")
        not in (None, architecture_state.state_id)
    ):
        raise CheckpointValidationError(
            "checkpoint_architecture_state_identity_mismatch",
            "architecture carry descriptor state identity does not match state",
        )
    if manifest is not None:
        architecture_manifest = manifest.get("architecture", {})
        if architecture_manifest.get("architecture_state_id") != (
            None if architecture_state is None else architecture_state.state_id
        ):
            raise CheckpointValidationError(
                "checkpoint_architecture_state_identity_mismatch",
                "architecture state identity does not match manifest",
            )
        if (
            architecture_manifest.get("parameter_catalog_digest")
            != parameter_catalog_digest
        ):
            raise CheckpointValidationError(
                "checkpoint_catalog_identity_mismatch",
                "parameter catalog identity does not match manifest",
            )
        if (
            architecture_manifest.get("architecture_config_digest")
            != architecture_config_digest
        ):
            raise CheckpointValidationError(
                "checkpoint_config_identity_mismatch",
                "architecture config identity does not match manifest",
            )
        if architecture_manifest.get("hf_reference") != hf_reference.to_dict():
            raise CheckpointValidationError(
                "checkpoint_hf_identity_mismatch",
                "HF lifecycle identity does not match manifest",
            )
        if architecture_manifest.get("architecture_carry_descriptor") != dict(
            architecture_carry_descriptor
        ):
            raise CheckpointValidationError(
                "checkpoint_architecture_descriptor_mismatch",
                "architecture carry descriptor does not match manifest",
            )
        if architecture_manifest.get("architecture_carry_identity_digest") != (
            descriptor_digest(architecture_carry_descriptor)
        ):
            raise CheckpointValidationError(
                "checkpoint_architecture_descriptor_hash_mismatch",
                "architecture carry descriptor identity hash mismatch",
            )


def _validate_carry_identity(
    identity: Mapping[str, Any],
    *,
    actual_descriptor: Mapping[str, Any],
    architecture_state: ArchitectureState | None,
) -> None:
    if identity.get("schema_version") != ARCHITECTURE_CARRY_SCHEMA_VERSION:
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_mismatch",
            "unsupported architecture carry descriptor schema",
        )
    if identity.get("pytree_descriptor_digest") != descriptor_digest(actual_descriptor):
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_mismatch",
            "architecture carry identity does not match its pytree descriptor",
        )
    if architecture_state is not None and identity.get("state_id") not in (
        None,
        architecture_state.state_id,
    ):
        raise CheckpointValidationError(
            "checkpoint_architecture_state_identity_mismatch",
            "architecture carry descriptor state identity does not match state",
        )


def _validate_expected_lifecycle_identity(
    *,
    hf_reference: HFPreservationReference,
    architecture_config_digest: str,
    parameter_catalog_digest: str,
    architecture_state: ArchitectureState | None,
    architecture_carry_descriptor: Mapping[str, Any],
    expected_hf_reference: HFPreservationReference | None,
    expected_architecture_config_digest: str | None,
    expected_parameter_catalog_digest: str | None,
    expected_architecture_state_id: str | None,
    expected_architecture_carry_descriptor: Mapping[str, Any] | None,
) -> None:
    if expected_hf_reference is not None and not isinstance(
        expected_hf_reference, HFPreservationReference
    ):
        raise TypeError("expected_hf_reference must be HFPreservationReference")
    if expected_hf_reference is not None and hf_reference != expected_hf_reference:
        raise CheckpointValidationError(
            "checkpoint_hf_identity_mismatch",
            "checkpoint HF identity does not match the requested resume identity",
        )
    if (
        expected_architecture_config_digest is not None
        and architecture_config_digest != expected_architecture_config_digest
    ):
        raise CheckpointValidationError(
            "checkpoint_config_identity_mismatch",
            "checkpoint architecture config identity does not match the "
            "requested resume identity",
        )
    if (
        expected_parameter_catalog_digest is not None
        and parameter_catalog_digest != expected_parameter_catalog_digest
    ):
        raise CheckpointValidationError(
            "checkpoint_catalog_identity_mismatch",
            "checkpoint parameter catalog identity does not match the "
            "requested resume identity",
        )
    if expected_architecture_state_id is not None and (
        architecture_state is None
        or architecture_state.state_id != expected_architecture_state_id
    ):
        raise CheckpointValidationError(
            "checkpoint_architecture_state_identity_mismatch",
            "checkpoint architecture state identity does not match the "
            "requested resume identity",
        )
    if expected_architecture_carry_descriptor is not None and dict(
        architecture_carry_descriptor
    ) != dict(expected_architecture_carry_descriptor):
        raise CheckpointValidationError(
            "checkpoint_architecture_descriptor_mismatch",
            "checkpoint architecture carry identity does not match the "
            "requested resume identity",
        )


def _fsync_tree(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optimizer_envelope(payload: Mapping[str, Any]) -> OptimizerState:
    return OptimizerState(
        optimizer_id=str(payload["optimizer_id"]),
        parameter_paths=tuple(payload["parameter_paths"]),
        step=int(payload["step"]),
        schema_version=str(payload["schema_version"]),
        state_structure=payload.get("state_structure", {}),
        backend_state=None,
        metadata=payload.get("metadata", {}),
        claims_not_made=tuple(payload.get("claims_not_made", ())),
    )


def _layout_from_dict(payload: Mapping[str, Any]) -> ParameterTreeLayout:
    entries = tuple(
        ParameterTreeLayoutEntry(
            logical_path=item["logical_path"],
            jax_keypath=tuple(item["jax_keypath"]),
            shape=tuple(item["shape"]),
            dtype=item["dtype"],
            role=item["role"],
            region_ids=tuple(item.get("region_ids", ())),
            trainable=bool(item.get("trainable", True)),
            exportable=bool(item.get("exportable", False)),
            hf_distribution_key=item.get("hf_distribution_key"),
            tied_weight_group=item.get("tied_weight_group"),
            metadata=item.get("metadata", {}),
        )
        for item in payload["entries"]
    )
    return ParameterTreeLayout(
        payload["architecture_id"], entries, payload["schema_version"]
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            "checkpoint_component_unreadable", "checkpoint JSON is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointValidationError(
            "checkpoint_component_invalid", "checkpoint JSON must be an object"
        )
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "CHECKPOINT_OPTIMIZER_STEP_MISMATCH",
    "CHECKPOINT_V3_SCHEMA_VERSION",
    "CheckpointValidationError",
    "JaxLearningCheckpointV3",
    "load_learning_checkpoint_v3",
    "optimizer_state_descriptor_payload",
    "save_learning_checkpoint_v3",
]
