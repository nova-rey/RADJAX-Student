"""Immutable, architecture-neutral models for the plugin boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.architecture._json import (
    freeze_mapping,
    json_value,
    mapping,
    nonempty_string,
    nonnegative_int,
    optional_string,
    strings,
)
from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
from radjax_student.contracts import (
    LearningBatch,
    ObjectiveScope,
    ResolvedObjectiveSelection,
)

ARCHITECTURE_CONFIG_SCHEMA_VERSION = "architecture_config.v1"
ARCHITECTURE_STATE_SCHEMA_VERSION = "architecture_state.v1"
PARAMETER_CATALOG_SCHEMA_VERSION = "parameter_catalog.v1"
ARCHITECTURE_PARAMETER_ROLES: tuple[str, ...] = (
    "adapter",
    "attention_block",
    "channel_mixer",
    "embedding",
    "normalization",
    "other",
    "output_head",
    "recurrent_block",
    "state_mixer",
)
ARCHITECTURE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "concrete_architecture_not_implemented",
    "parameter_initialization_not_executed_through_jax",
    "forward_execution_not_proven",
    "gradient_not_computed",
    "optimizer_not_invoked",
    "training_loop_not_run",
)


@dataclass(frozen=True)
class ArchitectureConfig:
    architecture_id: str
    schema_version: str = ARCHITECTURE_CONFIG_SCHEMA_VERSION
    model_config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    vocab_size: int | None = None
    sequence_length: int | None = None
    dtype_intent: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.architecture_id, "architecture_id")
        if self.schema_version != ARCHITECTURE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported architecture config schema version")
        for name in ("vocab_size", "sequence_length"):
            value = getattr(self, name)
            if value is not None:
                nonnegative_int(value, name)
                if value == 0:
                    raise ValueError(f"{name} must be positive when specified")
        nonempty_string(self.dtype_intent, "dtype_intent")
        object.__setattr__(self, "model_config", freeze_mapping(self.model_config))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "schema_version": self.schema_version,
            "model_config": json_value(self.model_config),
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "dtype_intent": self.dtype_intent,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureConfig:
        return cls(
            architecture_id=str(payload["architecture_id"]),
            schema_version=str(
                payload.get("schema_version", ARCHITECTURE_CONFIG_SCHEMA_VERSION)
            ),
            model_config=mapping(payload.get("model_config", {}), "model_config"),
            vocab_size=payload.get("vocab_size"),
            sequence_length=payload.get("sequence_length"),
            dtype_intent=str(payload.get("dtype_intent", "unspecified")),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ArchitectureCapabilityProfile:
    architecture_id: str
    version: int
    capabilities: tuple[str, ...]
    non_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.architecture_id, "architecture_id")
        nonnegative_int(self.version, "version")
        if self.version == 0:
            raise ValueError("version must be positive")
        capabilities = strings(self.capabilities, "capabilities", sort=True)
        non_capabilities = strings(self.non_capabilities, "non_capabilities", sort=True)
        overlap = sorted(set(capabilities) & set(non_capabilities))
        if overlap:
            raise ValueError("capabilities and non_capabilities cannot overlap")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "non_capabilities", non_capabilities)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "non_capabilities": list(self.non_capabilities),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureCapabilityProfile:
        return cls(
            architecture_id=str(payload["architecture_id"]),
            version=payload["version"],
            capabilities=strings(payload.get("capabilities", ()), "capabilities"),
            non_capabilities=strings(
                payload.get("non_capabilities", ()), "non_capabilities"
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ParameterDescriptor:
    path: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "other"
    region_ids: tuple[str, ...] = ()
    trainable_by_default: bool = True
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _stable_parameter_path(self.path)
        shape = tuple(self.shape)
        for dimension in shape:
            nonnegative_int(dimension, "shape dimension")
        nonempty_string(self.dtype, "dtype")
        if self.role not in ARCHITECTURE_PARAMETER_ROLES:
            raise ValueError("parameter role is unsupported")
        if not isinstance(self.trainable_by_default, bool):
            raise TypeError("trainable_by_default must be a boolean")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "region_ids", strings(self.region_ids, "region_ids", sort=True)
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
            "region_ids": list(self.region_ids),
            "trainable_by_default": self.trainable_by_default,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterDescriptor:
        return cls(
            path=str(payload["path"]),
            shape=tuple(payload["shape"]),
            dtype=str(payload["dtype"]),
            role=str(payload.get("role", "other")),
            region_ids=strings(payload.get("region_ids", ()), "region_ids"),
            trainable_by_default=payload.get("trainable_by_default", True),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ParameterCatalog:
    architecture_id: str
    parameters: tuple[ParameterDescriptor, ...]
    schema_version: str = PARAMETER_CATALOG_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.architecture_id, "architecture_id")
        if self.schema_version != PARAMETER_CATALOG_SCHEMA_VERSION:
            raise ValueError("unsupported parameter catalog schema version")
        parameters = tuple(self.parameters)
        if not parameters or any(
            not isinstance(item, ParameterDescriptor) for item in parameters
        ):
            raise TypeError("parameters must contain ParameterDescriptor values")
        paths = tuple(item.path for item in parameters)
        if len(paths) != len(set(paths)):
            raise ValueError("parameter catalog paths must be unique")
        object.__setattr__(
            self, "parameters", tuple(sorted(parameters, key=lambda item: item.path))
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.parameters)

    @property
    def trainable_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.parameters if item.trainable_by_default)

    def get(self, path: str) -> ParameterDescriptor:
        for descriptor in self.parameters:
            if descriptor.path == path:
                return descriptor
        raise KeyError(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "schema_version": self.schema_version,
            "parameters": [item.to_dict() for item in self.parameters],
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterCatalog:
        return cls(
            architecture_id=str(payload["architecture_id"]),
            schema_version=str(
                payload.get("schema_version", PARAMETER_CATALOG_SCHEMA_VERSION)
            ),
            parameters=tuple(
                ParameterDescriptor.from_dict(mapping(item, "parameter"))
                for item in payload["parameters"]
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class NamedRegion:
    region_id: str
    parameter_paths: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.region_id, "region_id")
        paths = strings(self.parameter_paths, "parameter_paths", sort=True)
        if not paths:
            raise ValueError("named region requires parameter paths")
        for path in paths:
            _stable_parameter_path(path)
        object.__setattr__(self, "parameter_paths", paths)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "parameter_paths": list(self.parameter_paths),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NamedRegion:
        return cls(
            region_id=str(payload["region_id"]),
            parameter_paths=strings(payload["parameter_paths"], "parameter_paths"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class IntermediateSurfaceDescriptor:
    surface_id: str
    kind: str
    shape_contract: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    region_id: str | None = None
    available_in_training: bool = False
    available_in_inference: bool = False
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.surface_id, "surface_id")
        nonempty_string(self.kind, "kind")
        if not isinstance(self.available_in_training, bool) or not isinstance(
            self.available_in_inference, bool
        ):
            raise TypeError("surface availability values must be booleans")
        object.__setattr__(
            self, "region_id", optional_string(self.region_id, "region_id")
        )
        object.__setattr__(self, "shape_contract", freeze_mapping(self.shape_contract))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "shape_contract": json_value(self.shape_contract),
            "region_id": self.region_id,
            "available_in_training": self.available_in_training,
            "available_in_inference": self.available_in_inference,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IntermediateSurfaceDescriptor:
        return cls(
            surface_id=str(payload["surface_id"]),
            kind=str(payload["kind"]),
            shape_contract=mapping(payload.get("shape_contract", {}), "shape_contract"),
            region_id=payload.get("region_id"),
            available_in_training=payload.get("available_in_training", False),
            available_in_inference=payload.get("available_in_inference", False),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ArchitectureMetadata:
    architecture_id: str
    parameter_catalog: ParameterCatalog
    capability_profile: ArchitectureCapabilityProfile
    named_regions: tuple[NamedRegion, ...] = ()
    objective_surfaces: tuple[IntermediateSurfaceDescriptor, ...] = ()
    warnings: tuple[ArchitectureIssue, ...] = ()
    claims_not_made: tuple[str, ...] = ARCHITECTURE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if (
            self.architecture_id != self.parameter_catalog.architecture_id
            or self.architecture_id != self.capability_profile.architecture_id
        ):
            raise ValueError("architecture metadata members must share architecture_id")
        regions = tuple(self.named_regions)
        surfaces = tuple(self.objective_surfaces)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, NamedRegion) for item in regions):
            raise TypeError("named_regions must contain NamedRegion values")
        if any(
            not isinstance(item, IntermediateSurfaceDescriptor) for item in surfaces
        ):
            raise TypeError(
                "objective_surfaces must contain IntermediateSurfaceDescriptor values"
            )
        if any(not isinstance(item, ArchitectureIssue) for item in warnings):
            raise TypeError("warnings must contain ArchitectureIssue values")
        if len({item.region_id for item in regions}) != len(regions):
            raise ValueError("named region IDs must be unique")
        if len({item.surface_id for item in surfaces}) != len(surfaces):
            raise ValueError("objective surface IDs must be unique")
        catalog_paths = set(self.parameter_catalog.paths)
        unknown = sorted(
            {path for item in regions for path in item.parameter_paths} - catalog_paths
        )
        if unknown:
            raise ValueError("named regions cannot reference unknown parameter paths")
        object.__setattr__(
            self,
            "named_regions",
            tuple(sorted(regions, key=lambda item: item.region_id)),
        )
        object.__setattr__(
            self,
            "objective_surfaces",
            tuple(sorted(surfaces, key=lambda item: item.surface_id)),
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_catalog.parameters)

    def region(self, region_id: str) -> NamedRegion:
        for region in self.named_regions:
            if region.region_id == region_id:
                return region
        raise KeyError(region_id)

    def surface(self, surface_id: str) -> IntermediateSurfaceDescriptor:
        for surface in self.objective_surfaces:
            if surface.surface_id == surface_id:
                return surface
        raise KeyError(surface_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "parameter_count": self.parameter_count,
            "parameter_catalog": self.parameter_catalog.to_dict(),
            "capability_profile": self.capability_profile.to_dict(),
            "named_regions": [item.to_dict() for item in self.named_regions],
            "objective_surfaces": [item.to_dict() for item in self.objective_surfaces],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureMetadata:
        return cls(
            architecture_id=str(payload["architecture_id"]),
            parameter_catalog=ParameterCatalog.from_dict(
                mapping(payload["parameter_catalog"], "parameter_catalog")
            ),
            capability_profile=ArchitectureCapabilityProfile.from_dict(
                mapping(payload["capability_profile"], "capability_profile")
            ),
            named_regions=tuple(
                NamedRegion.from_dict(mapping(item, "named_region"))
                for item in payload.get("named_regions", ())
            ),
            objective_surfaces=tuple(
                IntermediateSurfaceDescriptor.from_dict(
                    mapping(item, "objective_surface")
                )
                for item in payload.get("objective_surfaces", ())
            ),
            warnings=tuple(
                ArchitectureIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=strings(
                payload.get("claims_not_made", ARCHITECTURE_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class ArchitectureState:
    state_id: str
    schema_version: str = ARCHITECTURE_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.state_id, "state_id")
        if self.schema_version != ARCHITECTURE_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported architecture state schema version")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "schema_version": self.schema_version,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureState:
        return cls(
            state_id=str(payload["state_id"]),
            schema_version=str(
                payload.get("schema_version", ARCHITECTURE_STATE_SCHEMA_VERSION)
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ArchitectureInitRequest:
    config: ArchitectureConfig
    runtime_keys_reference: str
    precision_policy: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.config, ArchitectureConfig):
            raise TypeError("config must be ArchitectureConfig")
        nonempty_string(self.runtime_keys_reference, "runtime_keys_reference")
        nonempty_string(self.precision_policy, "precision_policy")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "runtime_keys_reference": self.runtime_keys_reference,
            "precision_policy": self.precision_policy,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureInitRequest:
        return cls(
            config=ArchitectureConfig.from_dict(mapping(payload["config"], "config")),
            runtime_keys_reference=str(payload["runtime_keys_reference"]),
            precision_policy=str(payload.get("precision_policy", "unspecified")),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ArchitectureInitResult:
    parameter_catalog: ParameterCatalog
    architecture_state: ArchitectureState | None = None
    parameters: Any = field(default=None, repr=False, compare=False)
    warnings: tuple[ArchitectureIssue, ...] = ()
    claims_not_made: tuple[str, ...] = ARCHITECTURE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_catalog, ParameterCatalog):
            raise TypeError("parameter_catalog must be ParameterCatalog")
        if self.architecture_state is not None and not isinstance(
            self.architecture_state, ArchitectureState
        ):
            raise TypeError(
                "architecture_state must be ArchitectureState when specified"
            )
        warnings = tuple(self.warnings)
        if any(not isinstance(item, ArchitectureIssue) for item in warnings):
            raise TypeError("warnings must contain ArchitectureIssue values")
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_catalog": self.parameter_catalog.to_dict(),
            "architecture_state": None
            if self.architecture_state is None
            else self.architecture_state.to_dict(),
            "parameters_present": self.parameters is not None,
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureInitResult:
        raw_state = payload.get("architecture_state")
        return cls(
            parameter_catalog=ParameterCatalog.from_dict(
                mapping(payload["parameter_catalog"], "parameter_catalog")
            ),
            architecture_state=(
                None
                if raw_state is None
                else ArchitectureState.from_dict(
                    mapping(raw_state, "architecture_state")
                )
            ),
            warnings=tuple(
                ArchitectureIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=strings(
                payload.get("claims_not_made", ARCHITECTURE_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class ForwardRequest:
    batch: LearningBatch
    objective_scope: ObjectiveScope = ObjectiveScope()
    architecture_state: ArchitectureState | None = None
    parameters: Any = field(default=None, repr=False, compare=False)
    training: bool = False
    rng_streams: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.batch, LearningBatch):
            raise TypeError("batch must be LearningBatch")
        if not isinstance(self.objective_scope, ObjectiveScope):
            raise TypeError("objective_scope must be ObjectiveScope")
        if self.architecture_state is not None and not isinstance(
            self.architecture_state, ArchitectureState
        ):
            raise TypeError(
                "architecture_state must be ArchitectureState when specified"
            )
        if not isinstance(self.training, bool):
            raise TypeError("training must be a boolean")
        object.__setattr__(self, "rng_streams", freeze_mapping(self.rng_streams))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch.to_dict(),
            "objective_scope": self.objective_scope.to_dict(),
            "architecture_state": None
            if self.architecture_state is None
            else self.architecture_state.to_dict(),
            "parameters_present": self.parameters is not None,
            "training": self.training,
            "rng_streams": json_value(self.rng_streams),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ForwardRequest:
        raw_state = payload.get("architecture_state")
        return cls(
            batch=LearningBatch.from_dict(mapping(payload["batch"], "batch")),
            objective_scope=ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
            architecture_state=(
                None
                if raw_state is None
                else ArchitectureState.from_dict(
                    mapping(raw_state, "architecture_state")
                )
            ),
            training=payload.get("training", False),
            rng_streams=mapping(payload.get("rng_streams", {}), "rng_streams"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ForwardResult:
    intermediate_surfaces: tuple[str, ...] = ()
    updated_architecture_state: ArchitectureState | None = None
    outputs: Any = field(default=None, repr=False, compare=False)
    surface_values: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    updated_architecture_carry: Any = field(default=None, repr=False, compare=False)
    architecture_metrics: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    output_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    warnings: tuple[ArchitectureIssue, ...] = ()
    claims_not_made: tuple[str, ...] = ARCHITECTURE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.updated_architecture_state is not None and not isinstance(
            self.updated_architecture_state, ArchitectureState
        ):
            raise TypeError(
                "updated_architecture_state must be ArchitectureState when specified"
            )
        warnings = tuple(self.warnings)
        if any(not isinstance(item, ArchitectureIssue) for item in warnings):
            raise TypeError("warnings must contain ArchitectureIssue values")
        object.__setattr__(
            self,
            "intermediate_surfaces",
            strings(self.intermediate_surfaces, "intermediate_surfaces", sort=True),
        )
        if not isinstance(self.surface_values, Mapping):
            raise TypeError("surface_values must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.surface_values):
            raise TypeError("surface_values keys must be nonempty strings")
        object.__setattr__(
            self, "surface_values", MappingProxyType(dict(self.surface_values))
        )
        if not isinstance(self.architecture_metrics, Mapping):
            raise TypeError("architecture_metrics must be a mapping")
        if any(
            not isinstance(key, str) or not key for key in self.architecture_metrics
        ):
            raise TypeError("architecture_metrics keys must be nonempty strings")
        object.__setattr__(
            self,
            "architecture_metrics",
            MappingProxyType(dict(self.architecture_metrics)),
        )
        object.__setattr__(
            self, "output_metadata", freeze_mapping(self.output_metadata)
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    def surface(self, surface_id: str) -> Any:
        """Return one architecture-owned objective surface value."""
        if surface_id == "final_output":
            if self.outputs is None:
                raise ArchitectureContractError(
                    "architecture_forward_failed",
                    "final_output surface is unavailable",
                )
            return self.outputs
        try:
            return self.surface_values[surface_id]
        except KeyError as exc:
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "requested forward surface is unavailable",
                details={"surface_id": surface_id},
            ) from exc

    def surface_for(self, selection: ResolvedObjectiveSelection | str) -> Any:
        """Return a surface already resolved by the owning architecture plugin."""

        if isinstance(selection, ResolvedObjectiveSelection):
            return self.surface(selection.surface_id)
        if isinstance(selection, str) and selection:
            return self.surface(selection)
        raise ArchitectureContractError(
            "architecture_objective_scope_resolution_failed",
            "forward surface requires a resolved objective selection or surface ID",
        )

    def surface_for_legacy_scope(self, scope: ObjectiveScope) -> Any:
        """Compatibility-only scope adapter retained for pre-P3.11 callers."""

        if not isinstance(scope, ObjectiveScope):
            raise ArchitectureContractError(
                "architecture_objective_scope_resolution_failed",
                "objective scope must be an ObjectiveScope",
            )
        if scope.kind in {"final_output", "whole_student"}:
            return self.surface("final_output")
        if scope.kind in {"intermediate_surface", "named_region", "plugin_defined"}:
            assert scope.target_id is not None
            return self.surface(scope.target_id)
        raise ArchitectureContractError(
            "architecture_objective_scope_resolution_failed",
            "legacy objective scope cannot be resolved to a forward surface",
            details={"kind": scope.kind},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs_present": self.outputs is not None,
            "intermediate_surfaces": list(self.intermediate_surfaces),
            "surface_ids": sorted(self.surface_values),
            "updated_architecture_carry_present": self.updated_architecture_carry
            is not None,
            "architecture_metric_names": sorted(self.architecture_metrics),
            "updated_architecture_state": None
            if self.updated_architecture_state is None
            else self.updated_architecture_state.to_dict(),
            "output_metadata": json_value(self.output_metadata),
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ForwardResult:
        raw_state = payload.get("updated_architecture_state")
        return cls(
            intermediate_surfaces=strings(
                payload.get("intermediate_surfaces", ()), "intermediate_surfaces"
            ),
            updated_architecture_state=(
                None
                if raw_state is None
                else ArchitectureState.from_dict(
                    mapping(raw_state, "updated_architecture_state")
                )
            ),
            output_metadata=mapping(
                payload.get("output_metadata", {}), "output_metadata"
            ),
            warnings=tuple(
                ArchitectureIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=strings(
                payload.get("claims_not_made", ARCHITECTURE_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class BatchValidationResult:
    status: Literal["pass", "fail"]
    blockers: tuple[ArchitectureIssue, ...] = ()
    warnings: tuple[ArchitectureIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("batch validation status must be pass or fail")
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        if any(
            not isinstance(item, ArchitectureIssue) for item in (*blockers, *warnings)
        ):
            raise TypeError(
                "batch validation findings must be ArchitectureIssue values"
            )
        if self.status == "pass" and blockers:
            raise ValueError("passing batch validation cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing batch validation requires blockers")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BatchValidationResult:
        return cls(
            status=str(payload["status"]),
            blockers=tuple(
                ArchitectureIssue.from_dict(mapping(item, "blocker"))
                for item in payload.get("blockers", ())
            ),
            warnings=tuple(
                ArchitectureIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


def canonical_architecture_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _stable_parameter_path(value: str) -> str:
    nonempty_string(value, "path")
    if (
        value != value.strip()
        or value.startswith(".")
        or value.endswith(".")
        or "/" in value
        or ".." in value
    ):
        raise ValueError("parameter path must be a stable dotted path")
    return value
