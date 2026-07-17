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
    HFCompatibilityDescriptor,
    HFPreservationReference,
    ObjectiveConfig,
    ObjectiveExecutionDescriptor,
    ObjectiveIdentity,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    ResolvedObjectiveSelection,
)
from radjax_student.learning.models import LearningState
from radjax_student.objectives.registry import ObjectiveRegistrySelection
from radjax_student.optimizers.jax import (
    JaxOptimizerState,
    validate_jax_optimizer_state,
)
from radjax_student.optimizers.models import OptimizerState
from radjax_student.optimizers.protocols import JaxOptimizerBackend

CHECKPOINT_V3_SCHEMA_VERSION = "learning_checkpoint.v3"
CHECKPOINT_OPTIMIZER_STEP_MISMATCH = "checkpoint_optimizer_step_mismatch"
CHECKPOINT_OBJECTIVE_IDENTITY_MISSING = "checkpoint_objective_identity_missing"
CHECKPOINT_HF_DESCRIPTOR_MISSING = "checkpoint_hf_descriptor_missing"
ARCHITECTURE_CARRY_SCHEMA_VERSION = "architecture_carry.v1"
V3_FILES = (
    "parameters.npz",
    "parameters.json",
    "architecture_carry.npz",
    "architecture_carry.json",
    "optimizer_state.npz",
    "optimizer_state.json",
    "hf_descriptor.json",
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
    "hf_descriptor.json": "architecture",
    "learning.json": "learning",
    "layout.json": "architecture",
}
HISTORICAL_MSE_OBJECTIVE_ALIASES = frozenset(
    {"mse", "linear.mse.v1", "stateful_linear_mse.v1"}
)
_HISTORICAL_MSE_IDENTITY = ObjectiveIdentity("radjax.objective.mean_squared_error", "1")


class CheckpointValidationError(ValueError):
    """Structured v3 validation failure with a stable blocker code."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.details = MappingProxyType(dict(details or {}))
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class HistoricalObjectiveMigration:
    """Inspection-only acknowledgement for exact pre-P3.12 MSE aliases.

    This record is deliberately not a lifecycle or checkpoint replacement.
    Continuation restore still requires the explicit canonical objective block.
    """

    source_alias: str
    canonical_objective_id: str
    canonical_objective_version: str
    status: str = "inspection_only_requires_recorded_migration"

    def __post_init__(self) -> None:
        if self.source_alias not in HISTORICAL_MSE_OBJECTIVE_ALIASES:
            raise CheckpointValidationError(
                CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
                "historical objective alias is not an accepted MSE migration source",
            )
        if (
            self.canonical_objective_id != _HISTORICAL_MSE_IDENTITY.objective_id
            or self.canonical_objective_version
            != _HISTORICAL_MSE_IDENTITY.objective_version
        ):
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "historical migration must target canonical MSE identity",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_alias": self.source_alias,
            "canonical_objective_id": self.canonical_objective_id,
            "canonical_objective_version": self.canonical_objective_version,
            "status": self.status,
        }


@dataclass(frozen=True)
class HistoricalHFDescriptorInspection:
    """Reference-only historical v3 inspection; never continuation state."""

    hf_reference: Mapping[str, Any]
    status: str = "inspection_only_descriptor_unavailable"

    def __post_init__(self) -> None:
        if not isinstance(self.hf_reference, Mapping):
            raise TypeError("historical HF inspection requires a serialized reference")
        if not self.hf_reference:
            raise CheckpointValidationError(
                CHECKPOINT_HF_DESCRIPTOR_MISSING,
                "historical checkpoint contains no preservation reference",
            )
        object.__setattr__(
            self, "hf_reference", MappingProxyType(dict(self.hf_reference))
        )

    def to_dict(self) -> dict[str, Any]:
        return {"hf_reference": dict(self.hf_reference), "status": self.status}


def inspect_historical_v3_hf_reference(
    directory: Path,
) -> HistoricalHFDescriptorInspection:
    """Read only a valid reference-only v3 shape without making it resumable."""

    if not directory.is_dir() or not (directory / "learning.json").is_file():
        raise CheckpointValidationError(
            "checkpoint_component_unreadable", "historical checkpoint is unreadable"
        )
    if (directory / "hf_descriptor.json").exists():
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "canonical descriptor checkpoint does not require historical inspection",
        )
    payload = _read_json(directory / "learning.json")
    return HistoricalHFDescriptorInspection(payload["hf_reference"])


def inspect_historical_v3_objective_alias(
    checkpoint: JaxLearningCheckpointV3, *, source_alias: str
) -> HistoricalObjectiveMigration:
    """Return explicit inspection migration evidence, never a resumable lifecycle."""

    if not isinstance(checkpoint, JaxLearningCheckpointV3):
        raise TypeError("historical objective inspection requires a v3 checkpoint")
    if checkpoint.objective_descriptor is not None:
        raise CheckpointValidationError(
            "checkpoint_objective_identity_mismatch",
            "canonical objective checkpoint does not require historical migration",
        )
    return HistoricalObjectiveMigration(
        source_alias=source_alias,
        canonical_objective_id=_HISTORICAL_MSE_IDENTITY.objective_id,
        canonical_objective_version=_HISTORICAL_MSE_IDENTITY.objective_version,
    )


@dataclass(frozen=True)
class JaxLearningCheckpointV3:
    runtime_reference: str
    learning_state: LearningState
    optimizer_state: JaxOptimizerState
    parameters: Mapping[str, Any]
    architecture_carry: Mapping[str, Any]
    parameter_layout: ParameterTreeLayout
    architecture_state: ArchitectureState | None
    hf_descriptor: HFCompatibilityDescriptor | None
    hf_reference: HFPreservationReference
    architecture_config_digest: str
    parameter_catalog_digest: str
    architecture_carry_descriptor: Mapping[str, Any] | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    integrity: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CHECKPOINT_V3_SCHEMA_VERSION
    objective_config: ObjectiveConfig | None = None
    resolved_objective_selection: ResolvedObjectiveSelection | None = None
    objective_descriptor: ObjectiveExecutionDescriptor | None = None
    objective_registry_selection: Mapping[str, Any] = field(default_factory=dict)

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
        if self.hf_descriptor is not None and not isinstance(
            self.hf_descriptor, HFCompatibilityDescriptor
        ):
            raise TypeError("hf_descriptor must be HFCompatibilityDescriptor")
        if not isinstance(self.hf_reference, HFPreservationReference):
            raise TypeError("hf_reference must be HFPreservationReference")
        if self.hf_descriptor is not None and (
            self.hf_reference != self.hf_descriptor.preservation_reference()
        ):
            raise ValueError("checkpoint HF reference must be descriptor-derived")
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
        if not isinstance(self.objective_registry_selection, Mapping):
            raise TypeError("objective_registry_selection must be a mapping")
        object.__setattr__(
            self,
            "objective_registry_selection",
            MappingProxyType(dict(self.objective_registry_selection)),
        )
        values = (
            self.objective_config,
            self.resolved_objective_selection,
            self.objective_descriptor,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise CheckpointValidationError(
                CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
                "objective checkpoint identity must be complete when present",
            )
        if all(value is not None for value in values):
            _validate_objective_identity(
                objective_config=self.objective_config,
                resolved_objective_selection=self.resolved_objective_selection,
                objective_descriptor=self.objective_descriptor,
                objective_registry_selection=self.objective_registry_selection,
                manifest=None,
            )


def save_learning_checkpoint_v3(
    checkpoint: JaxLearningCheckpointV3,
    directory: Path,
    *,
    optimizer: JaxOptimizerBackend,
) -> JaxLearningCheckpointV3:
    """Validate then write one canonical v3 continuation checkpoint."""

    if checkpoint.hf_descriptor is None:
        raise CheckpointValidationError(
            CHECKPOINT_HF_DESCRIPTOR_MISSING,
            "P3.12B checkpoint save requires a complete HF descriptor",
        )
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
            "hf_descriptor.json": checkpoint.hf_descriptor.to_dict(),
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
                "objective_config": checkpoint.objective_config.to_dict(),
                "resolved_objective_selection": (
                    checkpoint.resolved_objective_selection.to_dict()
                ),
                "objective_descriptor": checkpoint.objective_descriptor.to_dict(),
                "objective_registry_selection": dict(
                    checkpoint.objective_registry_selection
                ),
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
                "hf_descriptor_digest": checkpoint.hf_descriptor.digest,
                "tokenizer_identity_digest": (
                    checkpoint.hf_descriptor.tokenizer.digest
                ),
                "vocabulary_identity_digest": (
                    checkpoint.hf_descriptor.vocabulary.digest
                ),
                "special_token_identity_digest": (
                    checkpoint.hf_descriptor.special_tokens.digest
                ),
                "parameter_projection_digest": (
                    checkpoint.hf_descriptor.parameter_projection_digest
                ),
                "architecture_projection_digest": (
                    checkpoint.hf_descriptor.architecture_projection.digest
                ),
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
            "objective": {
                "descriptor": checkpoint.objective_descriptor.to_dict(),
                "descriptor_digest": checkpoint.objective_descriptor.digest,
                "config_digest": checkpoint.objective_config.digest,
                "resolved_surface_identity": (
                    checkpoint.resolved_objective_selection.digest
                ),
                "registry_selection": dict(checkpoint.objective_registry_selection),
            },
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
            expected_hf_reference=checkpoint.hf_reference,
            expected_hf_descriptor=checkpoint.hf_descriptor,
            expected_objective_descriptor=checkpoint.objective_descriptor,
            expected_objective_config=checkpoint.objective_config,
            expected_resolved_objective_selection=checkpoint.resolved_objective_selection,
            expected_objective_registry_selection=checkpoint.objective_registry_selection,
            require_objective_identity=False,
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
        runtime_reference=checkpoint.runtime_reference,
        learning_state=checkpoint.learning_state,
        optimizer_state=checkpoint.optimizer_state,
        parameters=checkpoint.parameters,
        architecture_carry=checkpoint.architecture_carry,
        parameter_layout=checkpoint.parameter_layout,
        architecture_state=checkpoint.architecture_state,
        hf_descriptor=checkpoint.hf_descriptor,
        hf_reference=checkpoint.hf_reference,
        architecture_config_digest=checkpoint.architecture_config_digest,
        parameter_catalog_digest=checkpoint.parameter_catalog_digest,
        architecture_carry_descriptor=architecture_carry_identity,
        manifest=manifest,
        integrity=integrity,
        objective_config=checkpoint.objective_config,
        resolved_objective_selection=checkpoint.resolved_objective_selection,
        objective_descriptor=checkpoint.objective_descriptor,
        objective_registry_selection=checkpoint.objective_registry_selection,
    )


def load_learning_checkpoint_v3(
    directory: Path,
    *,
    optimizer: JaxOptimizerBackend,
    parameter_layout: ParameterTreeLayout,
    runtime_reference: str | None = None,
    expected_hf_reference: HFPreservationReference | None = None,
    expected_hf_descriptor: HFCompatibilityDescriptor | None = None,
    expected_architecture_config_digest: str | None = None,
    expected_parameter_catalog_digest: str | None = None,
    expected_architecture_state_id: str | None = None,
    expected_architecture_carry_descriptor: Mapping[str, Any] | None = None,
    expected_objective_descriptor: ObjectiveExecutionDescriptor | None = None,
    expected_objective_config: ObjectiveConfig | None = None,
    expected_resolved_objective_selection: ResolvedObjectiveSelection | None = None,
    expected_objective_registry_selection: Mapping[str, Any] | None = None,
    expected_objective_selection: ObjectiveRegistrySelection | None = None,
    require_lifecycle_expectations: bool = False,
    require_objective_identity: bool = True,
) -> JaxLearningCheckpointV3:
    """Validate all v3 identity, integrity, and optimizer-owned invariants."""

    if require_lifecycle_expectations and any(
        value is None
        for value in (
            expected_hf_reference,
            expected_hf_descriptor,
            expected_architecture_config_digest,
            expected_parameter_catalog_digest,
            expected_architecture_carry_descriptor,
        )
    ):
        raise CheckpointValidationError(
            "checkpoint_lifecycle_expectations_missing",
            "caller-bound resume requires complete lifecycle expectations",
        )
    if require_objective_identity and any(
        value is None
        for value in (
            expected_objective_descriptor,
            expected_objective_config,
            expected_resolved_objective_selection,
            expected_objective_selection,
        )
    ):
        raise CheckpointValidationError(
            CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
            "caller-bound objective restore requires complete expected "
            "objective identity",
        )
    if expected_objective_selection is not None:
        if not isinstance(expected_objective_selection, ObjectiveRegistrySelection):
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "expected objective selection must come from ObjectiveRegistry",
            )
        if not expected_objective_selection.is_registry_selected:
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "expected objective selection was not issued by ObjectiveRegistry",
            )
        selected = expected_objective_selection.to_dict()
        if (
            expected_objective_descriptor is not None
            and expected_objective_descriptor.implementation_identity
            != expected_objective_selection.implementation_identity
        ):
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "caller objective descriptor does not match selected plugin",
            )
        if (
            expected_objective_registry_selection is not None
            and dict(expected_objective_registry_selection) != selected
        ):
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "caller objective registry selection identities disagree",
            )
        expected_objective_registry_selection = selected
    if not directory.is_dir():
        raise CheckpointValidationError(
            "checkpoint_component_unreadable", "checkpoint directory is unreadable"
        )
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    expected_files = set(V3_FILES)
    unexpected_files = actual_files - expected_files
    if unexpected_files:
        raise CheckpointValidationError(
            "checkpoint_unexpected_file",
            "checkpoint contains undeclared files",
            details={"files": sorted(unexpected_files)},
        )
    stored = _read_json(directory / "manifest.json")
    integrity = stored.pop("integrity", None)
    if not isinstance(integrity, Mapping) or integrity.get(
        "manifest_digest"
    ) != _digest(_json_bytes(stored)):
        raise CheckpointValidationError(
            "checkpoint_manifest_hash_mismatch", "checkpoint manifest hash mismatch"
        )
    # Historical v3 checkpoints had a valid manifest but no complete HF
    # descriptor.  Classify them before modern exact-file validation.
    if (
        "hf_descriptor.json" not in stored.get("files", ())
        or not (directory / "hf_descriptor.json").is_file()
    ):
        raise CheckpointValidationError(
            CHECKPOINT_HF_DESCRIPTOR_MISSING,
            "checkpoint predates canonical HF descriptor and is inspection-only",
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
    hf_descriptor = HFCompatibilityDescriptor.from_dict(
        _read_json(directory / "hf_descriptor.json")
    )
    if hf_reference != hf_descriptor.preservation_reference():
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF reference is not derived from its descriptor",
        )
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
    objective_config_payload = learning_payload.get("objective_config")
    resolved_objective_payload = learning_payload.get("resolved_objective_selection")
    objective_descriptor_payload = learning_payload.get("objective_descriptor")
    objective_registry_selection = learning_payload.get("objective_registry_selection")
    objective_values = (
        objective_config_payload,
        resolved_objective_payload,
        objective_descriptor_payload,
        objective_registry_selection,
    )
    if any(value is not None for value in objective_values) and not all(
        value is not None for value in objective_values
    ):
        raise CheckpointValidationError(
            CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
            "checkpoint objective identity block is incomplete",
        )
    if not all(value is not None for value in objective_values):
        if require_objective_identity:
            raise CheckpointValidationError(
                CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
                "checkpoint predates canonical objective identity and is "
                "inspection-only",
            )
        objective_config = None
        resolved_objective_selection = None
        objective_descriptor = None
        objective_registry_selection = MappingProxyType({})
    else:
        if not isinstance(objective_registry_selection, Mapping):
            raise CheckpointValidationError(
                CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
                "checkpoint objective registry selection is invalid",
            )
        objective_config = ObjectiveConfig.from_dict(objective_config_payload)
        resolved_objective_selection = ResolvedObjectiveSelection.from_dict(
            resolved_objective_payload
        )
        objective_descriptor = ObjectiveExecutionDescriptor.from_dict(
            objective_descriptor_payload
        )
        _validate_objective_identity(
            objective_config=objective_config,
            resolved_objective_selection=resolved_objective_selection,
            objective_descriptor=objective_descriptor,
            objective_registry_selection=objective_registry_selection,
            manifest=stored,
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
    if expected_hf_descriptor is None:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "caller-bound continuation requires expected HF descriptor",
        )
    if not isinstance(expected_hf_descriptor, HFCompatibilityDescriptor):
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "expected HF descriptor is invalid",
        )
    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )
    if objective_descriptor is not None:
        _validate_expected_objective_identity(
            objective_config=objective_config,
            resolved_objective_selection=resolved_objective_selection,
            objective_descriptor=objective_descriptor,
            objective_registry_selection=objective_registry_selection,
            expected_objective_descriptor=expected_objective_descriptor,
            expected_objective_config=expected_objective_config,
            expected_resolved_objective_selection=expected_resolved_objective_selection,
            expected_objective_registry_selection=expected_objective_registry_selection,
        )
    return JaxLearningCheckpointV3(
        runtime_reference=learning_payload["runtime_reference"],
        learning_state=learning_state,
        optimizer_state=state,
        parameters=parameters,
        architecture_carry=carry,
        parameter_layout=parameter_layout,
        architecture_state=architecture_state,
        hf_descriptor=hf_descriptor,
        hf_reference=hf_reference,
        architecture_config_digest=architecture_config_digest,
        parameter_catalog_digest=parameter_catalog_digest,
        architecture_carry_descriptor=architecture_carry_descriptor,
        manifest=stored,
        integrity=integrity,
        objective_config=objective_config,
        resolved_objective_selection=resolved_objective_selection,
        objective_descriptor=objective_descriptor,
        objective_registry_selection=objective_registry_selection,
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
    if checkpoint.hf_descriptor is None:
        raise CheckpointValidationError(
            CHECKPOINT_HF_DESCRIPTOR_MISSING,
            "continuation checkpoint requires a complete HF descriptor",
        )
    if checkpoint.hf_reference != checkpoint.hf_descriptor.preservation_reference():
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF reference was not derived from its descriptor",
        )
    if (
        checkpoint.objective_config is None
        or checkpoint.resolved_objective_selection is None
        or checkpoint.objective_descriptor is None
        or not checkpoint.objective_registry_selection
    ):
        raise CheckpointValidationError(
            CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
            "continuation checkpoint requires canonical objective identity",
        )
    _validate_objective_identity(
        objective_config=checkpoint.objective_config,
        resolved_objective_selection=checkpoint.resolved_objective_selection,
        objective_descriptor=checkpoint.objective_descriptor,
        objective_registry_selection=checkpoint.objective_registry_selection,
        manifest=None,
    )
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


def _validate_objective_identity(
    *,
    objective_config: ObjectiveConfig,
    resolved_objective_selection: ResolvedObjectiveSelection,
    objective_descriptor: ObjectiveExecutionDescriptor,
    objective_registry_selection: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    if (
        not isinstance(objective_config, ObjectiveConfig)
        or not isinstance(resolved_objective_selection, ResolvedObjectiveSelection)
        or not isinstance(objective_descriptor, ObjectiveExecutionDescriptor)
    ):
        raise CheckpointValidationError(
            CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
            "checkpoint objective identity contracts are malformed",
        )
    required_selection = {
        "objective_id",
        "objective_version",
        "capability_profile_digest",
        "implementation_identity",
        "registry_identity",
    }
    if set(objective_registry_selection) != required_selection:
        raise CheckpointValidationError(
            "checkpoint_objective_identity_mismatch",
            "checkpoint objective registry selection fields are invalid",
        )
    expected = {
        "objective_id": objective_descriptor.identity.objective_id,
        "objective_version": objective_descriptor.identity.objective_version,
        "capability_profile_digest": objective_descriptor.capability_profile_digest,
        "implementation_identity": objective_descriptor.implementation_identity,
    }
    if any(
        objective_registry_selection[name] != value for name, value in expected.items()
    ):
        raise CheckpointValidationError(
            "checkpoint_objective_identity_mismatch",
            "checkpoint objective registry selection disagrees with descriptor",
        )
    if (
        objective_config.identity != objective_descriptor.identity
        or objective_config.digest != objective_descriptor.config_digest
        or resolved_objective_selection.digest
        != objective_descriptor.resolved_surface_identity
    ):
        raise CheckpointValidationError(
            "checkpoint_objective_identity_mismatch",
            "checkpoint objective config, surface, and descriptor disagree",
        )
    if manifest is not None:
        objective_manifest = manifest.get("objective")
        if not isinstance(objective_manifest, Mapping):
            raise CheckpointValidationError(
                CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
                "checkpoint manifest lacks canonical objective identity",
            )
        if (
            objective_manifest.get("descriptor") != objective_descriptor.to_dict()
            or objective_manifest.get("descriptor_digest")
            != objective_descriptor.digest
            or objective_manifest.get("config_digest") != objective_config.digest
            or objective_manifest.get("resolved_surface_identity")
            != resolved_objective_selection.digest
            or objective_manifest.get("registry_selection")
            != dict(objective_registry_selection)
        ):
            raise CheckpointValidationError(
                "checkpoint_objective_identity_mismatch",
                "checkpoint manifest objective identity disagrees with "
                "learning payload",
            )


def _validate_expected_objective_identity(
    *,
    objective_config: ObjectiveConfig,
    resolved_objective_selection: ResolvedObjectiveSelection,
    objective_descriptor: ObjectiveExecutionDescriptor,
    objective_registry_selection: Mapping[str, Any],
    expected_objective_descriptor: ObjectiveExecutionDescriptor | None,
    expected_objective_config: ObjectiveConfig | None,
    expected_resolved_objective_selection: ResolvedObjectiveSelection | None,
    expected_objective_registry_selection: Mapping[str, Any] | None,
) -> None:
    expected_values = (
        expected_objective_descriptor,
        expected_objective_config,
        expected_resolved_objective_selection,
        expected_objective_registry_selection,
    )
    if any(value is None for value in expected_values):
        return
    if (
        not isinstance(expected_objective_descriptor, ObjectiveExecutionDescriptor)
        or not isinstance(expected_objective_config, ObjectiveConfig)
        or not isinstance(
            expected_resolved_objective_selection, ResolvedObjectiveSelection
        )
        or not isinstance(expected_objective_registry_selection, Mapping)
    ):
        raise TypeError("expected objective identity contracts are invalid")
    if (
        objective_descriptor != expected_objective_descriptor
        or objective_config != expected_objective_config
        or resolved_objective_selection != expected_resolved_objective_selection
        or dict(objective_registry_selection)
        != dict(expected_objective_registry_selection)
    ):
        raise CheckpointValidationError(
            "checkpoint_objective_identity_mismatch",
            "checkpoint objective identity does not match caller-selected objective",
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
    "CHECKPOINT_OBJECTIVE_IDENTITY_MISSING",
    "CHECKPOINT_OPTIMIZER_STEP_MISMATCH",
    "CHECKPOINT_V3_SCHEMA_VERSION",
    "CheckpointValidationError",
    "HistoricalObjectiveMigration",
    "HISTORICAL_MSE_OBJECTIVE_ALIASES",
    "JaxLearningCheckpointV3",
    "load_learning_checkpoint_v3",
    "inspect_historical_v3_objective_alias",
    "optimizer_state_descriptor_payload",
    "save_learning_checkpoint_v3",
]
