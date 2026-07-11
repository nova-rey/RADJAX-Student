from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from radjax_student.runtime.keys import RuntimeKeys

PrecisionPolicy: TypeAlias = Literal[
    "float32",
    "bfloat16",
    "float16",
    "mixed",
    "automatic",
    "unspecified",
]
PlacementPolicy: TypeAlias = Literal[
    "single_device",
    "replicated",
    "data_sharded",
    "model_sharded",
    "automatic",
    "unspecified",
]
CompilationPolicy: TypeAlias = Literal["eager", "jit", "automatic", "unspecified"]
DistributedPolicy: TypeAlias = Literal["disabled", "auto", "required"]
FallbackPolicy: TypeAlias = Literal["disallowed", "allow_compatible"]
RuntimeStatus: TypeAlias = Literal["pass", "fail"]

PRECISION_POLICIES: tuple[str, ...] = (
    "float32",
    "bfloat16",
    "float16",
    "mixed",
    "automatic",
    "unspecified",
)
PLACEMENT_POLICIES: tuple[str, ...] = (
    "single_device",
    "replicated",
    "data_sharded",
    "model_sharded",
    "automatic",
    "unspecified",
)
COMPILATION_POLICIES: tuple[str, ...] = (
    "eager",
    "jit",
    "automatic",
    "unspecified",
)
DISTRIBUTED_POLICIES: tuple[str, ...] = ("disabled", "auto", "required")
FALLBACK_POLICIES: tuple[str, ...] = ("disallowed", "allow_compatible")
RUNTIME_CAPABILITY_VOCABULARY: tuple[str, ...] = (
    "compilation.jit_v1",
    "execution.argument_donation_v1",
    "execution.eager_v1",
    "execution.static_arguments_v1",
    "execution.synchronize_v1",
    "placement.data_sharded_v1",
    "placement.model_sharded_v1",
    "placement.replicated_v1",
    "placement.single_device_v1",
    "runtime.multi_process_v1",
    "runtime.single_process_v1",
    "state.runtime_envelope_v1",
)
RUNTIME_STATE_SCHEMA_VERSION = "runtime_state.v1"


@dataclass(frozen=True)
class RuntimeConfig:
    """Requested runtime intent, distinct from observed environment facts."""

    backend_id: str | None = None
    platform_preference: str = "unspecified"
    precision_policy: PrecisionPolicy = "unspecified"
    placement_policy: PlacementPolicy = "unspecified"
    compilation_policy: CompilationPolicy = "unspecified"
    distributed_policy: DistributedPolicy = "disabled"
    fallback_policy: FallbackPolicy = "disallowed"
    required_capabilities: tuple[str, ...] = ()
    seed: int = 0
    debug: bool = False

    def __post_init__(self) -> None:
        _require_optional_string("backend_id", self.backend_id)
        if self.backend_id == "":
            raise ValueError("backend_id must be nonempty when specified")
        _require_string("platform_preference", self.platform_preference)
        if not self.platform_preference:
            raise ValueError("platform_preference must be nonempty")
        _require_choice("precision_policy", self.precision_policy, PRECISION_POLICIES)
        _require_choice("placement_policy", self.placement_policy, PLACEMENT_POLICIES)
        _require_choice(
            "compilation_policy",
            self.compilation_policy,
            COMPILATION_POLICIES,
        )
        _require_choice(
            "distributed_policy",
            self.distributed_policy,
            DISTRIBUTED_POLICIES,
        )
        _require_choice("fallback_policy", self.fallback_policy, FALLBACK_POLICIES)
        object.__setattr__(
            self,
            "required_capabilities",
            _unique_sorted_strings(
                self.required_capabilities,
                "required_capabilities",
            ),
        )
        _require_integer("seed", self.seed)
        _require_boolean("debug", self.debug)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "platform_preference": self.platform_preference,
            "precision_policy": self.precision_policy,
            "placement_policy": self.placement_policy,
            "compilation_policy": self.compilation_policy,
            "distributed_policy": self.distributed_policy,
            "fallback_policy": self.fallback_policy,
            "required_capabilities": list(self.required_capabilities),
            "seed": self.seed,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeConfig:
        return cls(
            backend_id=_optional_string(payload.get("backend_id")),
            platform_preference=str(payload.get("platform_preference", "unspecified")),
            precision_policy=str(payload.get("precision_policy", "unspecified")),
            placement_policy=str(payload.get("placement_policy", "unspecified")),
            compilation_policy=str(payload.get("compilation_policy", "unspecified")),
            distributed_policy=str(payload.get("distributed_policy", "disabled")),
            fallback_policy=str(payload.get("fallback_policy", "disallowed")),
            required_capabilities=_string_tuple(
                payload.get("required_capabilities", ()),
                "required_capabilities",
            ),
            seed=_required_int(payload.get("seed", 0), "seed"),
            debug=_required_bool(payload.get("debug", False), "debug"),
        )


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Observed process and runtime facts; unknown values remain ``None``."""

    python_version: str
    jax_available: bool
    jax_version: str | None = None
    jaxlib_version: str | None = None
    platform: str | None = None
    process_count: int | None = None
    process_index: int | None = None
    local_device_count: int | None = None
    global_device_count: int | None = None
    distributed_initialized: bool | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string("python_version", self.python_version)
        if not self.python_version:
            raise ValueError("python_version must be nonempty")
        _require_boolean("jax_available", self.jax_available)
        for name in ("jax_version", "jaxlib_version", "platform"):
            _require_optional_string(name, getattr(self, name))
        for name in (
            "process_count",
            "process_index",
            "local_device_count",
            "global_device_count",
        ):
            value = getattr(self, name)
            _require_optional_integer(name, value)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when observed")
        _require_optional_boolean(
            "distributed_initialized",
            self.distributed_initialized,
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "jax_available": self.jax_available,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "platform": self.platform,
            "process_count": self.process_count,
            "process_index": self.process_index,
            "local_device_count": self.local_device_count,
            "global_device_count": self.global_device_count,
            "distributed_initialized": self.distributed_initialized,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeEnvironment:
        return cls(
            python_version=str(payload["python_version"]),
            jax_available=_required_bool(payload["jax_available"], "jax_available"),
            jax_version=_optional_string(payload.get("jax_version")),
            jaxlib_version=_optional_string(payload.get("jaxlib_version")),
            platform=_optional_string(payload.get("platform")),
            process_count=_optional_int(payload.get("process_count")),
            process_index=_optional_int(payload.get("process_index")),
            local_device_count=_optional_int(payload.get("local_device_count")),
            global_device_count=_optional_int(payload.get("global_device_count")),
            distributed_initialized=_optional_bool(
                payload.get("distributed_initialized")
            ),
            warnings=_string_tuple(payload.get("warnings", ()), "warnings"),
        )


@dataclass(frozen=True)
class DeviceDescriptor:
    """Serializable device facts without retaining a raw backend device object."""

    device_id: str
    platform: str | None = None
    device_kind: str | None = None
    process_index: int | None = None
    local_hardware_id: str | int | None = None
    memory_bytes: int | None = None
    supported_precisions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _require_string("device_id", self.device_id)
        if not self.device_id:
            raise ValueError("device_id must be nonempty")
        _require_optional_string("platform", self.platform)
        _require_optional_string("device_kind", self.device_kind)
        _require_optional_integer("process_index", self.process_index)
        if self.process_index is not None and self.process_index < 0:
            raise ValueError("process_index must be nonnegative when observed")
        if self.local_hardware_id is not None and (
            isinstance(self.local_hardware_id, bool)
            or not isinstance(self.local_hardware_id, (str, int))
        ):
            raise TypeError("local_hardware_id must be a string or integer")
        _require_optional_integer("memory_bytes", self.memory_bytes)
        if self.memory_bytes is not None and self.memory_bytes < 0:
            raise ValueError("memory_bytes must be nonnegative when observed")
        object.__setattr__(
            self,
            "supported_precisions",
            _string_tuple(self.supported_precisions, "supported_precisions"),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "platform": self.platform,
            "device_kind": self.device_kind,
            "process_index": self.process_index,
            "local_hardware_id": self.local_hardware_id,
            "memory_bytes": self.memory_bytes,
            "supported_precisions": list(self.supported_precisions),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeviceDescriptor:
        return cls(
            device_id=str(payload["device_id"]),
            platform=_optional_string(payload.get("platform")),
            device_kind=_optional_string(payload.get("device_kind")),
            process_index=_optional_int(payload.get("process_index")),
            local_hardware_id=payload.get("local_hardware_id"),
            memory_bytes=_optional_int(payload.get("memory_bytes")),
            supported_precisions=_string_tuple(
                payload.get("supported_precisions", ()),
                "supported_precisions",
            ),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class DeviceInventory:
    """Immutable normalized devices and optional topology summary."""

    devices: tuple[DeviceDescriptor, ...] = ()
    process_count: int | None = None
    local_device_count: int | None = None
    global_device_count: int | None = None
    topology_summary: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        devices = tuple(self.devices)
        if any(not isinstance(item, DeviceDescriptor) for item in devices):
            raise TypeError("devices must contain DeviceDescriptor values")
        identifiers = [item.device_id for item in devices]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("device IDs must be unique")
        for name in ("process_count", "local_device_count", "global_device_count"):
            value = getattr(self, name)
            _require_optional_integer(name, value)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when observed")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(
            self,
            "topology_summary",
            freeze_json_mapping(self.topology_summary),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "process_count": self.process_count,
            "local_device_count": self.local_device_count,
            "global_device_count": self.global_device_count,
            "topology_summary": json_value(self.topology_summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeviceInventory:
        raw_devices = payload.get("devices", ())
        if not isinstance(raw_devices, (list, tuple)):
            raise TypeError("devices must be a list or tuple")
        return cls(
            devices=tuple(
                DeviceDescriptor.from_dict(_mapping(item, "device"))
                for item in raw_devices
            ),
            process_count=_optional_int(payload.get("process_count")),
            local_device_count=_optional_int(payload.get("local_device_count")),
            global_device_count=_optional_int(payload.get("global_device_count")),
            topology_summary=_mapping(
                payload.get("topology_summary", {}),
                "topology_summary",
            ),
        )


@dataclass(frozen=True)
class RuntimeCapabilityProfile:
    """Versioned backend declarations, never execution proof."""

    profile_id: str
    backend_id: str
    version: int
    capabilities: tuple[str, ...] = ()
    non_capabilities: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string("profile_id", self.profile_id)
        _require_string("backend_id", self.backend_id)
        if not self.profile_id or not self.backend_id:
            raise ValueError("profile_id and backend_id must be nonempty")
        _require_integer("version", self.version)
        if self.version <= 0:
            raise ValueError("capability profile version must be positive")
        capabilities = _unique_sorted_strings(self.capabilities, "capabilities")
        non_capabilities = _unique_sorted_strings(
            self.non_capabilities,
            "non_capabilities",
        )
        overlap = set(capabilities) & set(non_capabilities)
        if overlap:
            raise ValueError(
                "capabilities and non_capabilities overlap: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "non_capabilities", non_capabilities)
        object.__setattr__(self, "notes", _string_tuple(self.notes, "notes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "backend_id": self.backend_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "non_capabilities": list(self.non_capabilities),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeCapabilityProfile:
        return cls(
            profile_id=str(payload["profile_id"]),
            backend_id=str(payload["backend_id"]),
            version=_required_int(payload["version"], "version"),
            capabilities=_string_tuple(payload.get("capabilities", ()), "capabilities"),
            non_capabilities=_string_tuple(
                payload.get("non_capabilities", ()),
                "non_capabilities",
            ),
            notes=_string_tuple(payload.get("notes", ()), "notes"),
        )


@dataclass(frozen=True)
class CompilationOptions:
    """Stable execution policy, intentionally smaller than raw ``jax.jit``."""

    enabled: bool = False
    mode: CompilationPolicy = "eager"
    static_arg_names: tuple[str, ...] = ()
    static_arg_positions: tuple[int, ...] = ()
    donate_arg_names: tuple[str, ...] = ()
    donate_arg_positions: tuple[int, ...] = ()
    debug: bool = False
    synchronize_results: bool = False
    cache_policy: Literal["reuse", "disabled"] = "reuse"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        for name in ("enabled", "debug", "synchronize_results"):
            _require_boolean(name, getattr(self, name))
        _require_choice("mode", self.mode, ("eager", "jit", "automatic"))
        mode = "jit" if self.enabled and self.mode == "eager" else self.mode
        if self.enabled and mode != "jit":
            raise ValueError("enabled compilation compatibility flag requires jit mode")
        if not self.enabled and self.mode == "jit":
            object.__setattr__(self, "enabled", True)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "static_arg_names",
            _unique_strings(self.static_arg_names, "static_arg_names"),
        )
        object.__setattr__(
            self,
            "static_arg_positions",
            _unique_nonnegative_integers(
                self.static_arg_positions,
                "static_arg_positions",
            ),
        )
        object.__setattr__(
            self,
            "donate_arg_names",
            _unique_strings(self.donate_arg_names, "donate_arg_names"),
        )
        object.__setattr__(
            self,
            "donate_arg_positions",
            _unique_nonnegative_integers(
                self.donate_arg_positions,
                "donate_arg_positions",
            ),
        )
        if set(self.static_arg_names) & set(self.donate_arg_names):
            raise ValueError("static and donated argument names must not overlap")
        if set(self.static_arg_positions) & set(self.donate_arg_positions):
            raise ValueError("static and donated argument positions must not overlap")
        _require_choice("cache_policy", self.cache_policy, ("reuse", "disabled"))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("compilation metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "static_arg_names": list(self.static_arg_names),
            "static_arg_positions": list(self.static_arg_positions),
            "donate_arg_names": list(self.donate_arg_names),
            "donate_arg_positions": list(self.donate_arg_positions),
            "debug": self.debug,
            "synchronize_results": self.synchronize_results,
            "cache_policy": self.cache_policy,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompilationOptions:
        return cls(
            enabled=_required_bool(payload.get("enabled", False), "enabled"),
            mode=str(
                payload.get(
                    "mode",
                    "jit" if payload.get("enabled", False) else "eager",
                )
            ),
            static_arg_names=_string_tuple(
                payload.get("static_arg_names", ()),
                "static_arg_names",
            ),
            static_arg_positions=_integer_tuple(
                payload.get("static_arg_positions", ()),
                "static_arg_positions",
            ),
            donate_arg_names=_string_tuple(
                payload.get("donate_arg_names", ()),
                "donate_arg_names",
            ),
            donate_arg_positions=_integer_tuple(
                payload.get("donate_arg_positions", ()),
                "donate_arg_positions",
            ),
            debug=_required_bool(payload.get("debug", False), "debug"),
            synchronize_results=_required_bool(
                payload.get("synchronize_results", False),
                "synchronize_results",
            ),
            cache_policy=str(payload.get("cache_policy", "reuse")),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ExecutionContext:
    """Initialized runtime-owned context; never model or optimizer state."""

    backend_id: str
    environment: RuntimeEnvironment
    device_inventory: DeviceInventory
    capabilities: RuntimeCapabilityProfile
    root_seed: int
    runtime_id: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    runtime_keys: RuntimeKeys | None = None

    def __post_init__(self) -> None:
        _require_string("backend_id", self.backend_id)
        _require_string("runtime_id", self.runtime_id)
        if not self.backend_id or not self.runtime_id:
            raise ValueError("backend_id and runtime_id must be nonempty")
        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be RuntimeEnvironment")
        if not isinstance(self.device_inventory, DeviceInventory):
            raise TypeError("device_inventory must be DeviceInventory")
        if not isinstance(self.capabilities, RuntimeCapabilityProfile):
            raise TypeError("capabilities must be RuntimeCapabilityProfile")
        if self.backend_id != self.capabilities.backend_id:
            raise ValueError("context backend and capability backend must match")
        _require_integer("root_seed", self.root_seed)
        if self.root_seed < 0:
            raise ValueError("root_seed must be nonnegative")
        runtime_keys = (
            RuntimeKeys.from_seed(self.root_seed)
            if self.runtime_keys is None
            else self.runtime_keys
        )
        if not isinstance(runtime_keys, RuntimeKeys):
            raise TypeError("runtime_keys must be RuntimeKeys when specified")
        if runtime_keys.root_seed != self.root_seed:
            raise ValueError("runtime key root seed must match context root_seed")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        object.__setattr__(self, "runtime_keys", runtime_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "environment": self.environment.to_dict(),
            "device_inventory": self.device_inventory.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "root_seed": self.root_seed,
            "runtime_id": self.runtime_id,
            "metadata": json_value(self.metadata),
            "runtime_keys": self.runtime_keys.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionContext:
        return cls(
            backend_id=str(payload["backend_id"]),
            environment=RuntimeEnvironment.from_dict(
                _mapping(payload["environment"], "environment")
            ),
            device_inventory=DeviceInventory.from_dict(
                _mapping(payload["device_inventory"], "device_inventory")
            ),
            capabilities=RuntimeCapabilityProfile.from_dict(
                _mapping(payload["capabilities"], "capabilities")
            ),
            root_seed=_required_int(payload["root_seed"], "root_seed"),
            runtime_id=str(payload["runtime_id"]),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
            runtime_keys=(
                None
                if payload.get("runtime_keys") is None
                else RuntimeKeys.from_dict(
                    _mapping(payload["runtime_keys"], "runtime_keys")
                )
            ),
        )


@dataclass(frozen=True)
class RuntimeState:
    """Portable, versioned runtime-only state with no executable or model data."""

    runtime_id: str
    global_step: int
    root_seed: int
    runtime_config: RuntimeConfig
    topology_summary: Mapping[str, Any]
    precision_policy: PrecisionPolicy
    placement_policy: PlacementPolicy
    resume_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = RUNTIME_STATE_SCHEMA_VERSION
    runtime_keys: RuntimeKeys | None = None
    environment_summary: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    backend_id: str | None = None
    claims_not_made: tuple[str, ...] = (
        "model_parameters_not_included",
        "optimizer_state_not_included",
        "compiled_executables_not_included",
        "architecture_state_not_included",
    )

    def __post_init__(self) -> None:
        _require_string("runtime_id", self.runtime_id)
        if not self.runtime_id:
            raise ValueError("runtime_id must be nonempty")
        _require_integer("global_step", self.global_step)
        _require_integer("root_seed", self.root_seed)
        if not isinstance(self.runtime_config, RuntimeConfig):
            raise TypeError("runtime_config must be RuntimeConfig")
        if self.global_step < 0 or self.root_seed < 0:
            raise ValueError("global_step and root_seed must be nonnegative")
        if self.schema_version != RUNTIME_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported runtime state schema version")
        _require_choice("precision_policy", self.precision_policy, PRECISION_POLICIES)
        _require_choice("placement_policy", self.placement_policy, PLACEMENT_POLICIES)
        runtime_keys = (
            RuntimeKeys.from_seed(self.root_seed)
            if self.runtime_keys is None
            else self.runtime_keys
        )
        if not isinstance(runtime_keys, RuntimeKeys):
            raise TypeError("runtime_keys must be RuntimeKeys when specified")
        if runtime_keys.root_seed != self.root_seed:
            raise ValueError("runtime key root seed must match state root_seed")
        _require_optional_string("backend_id", self.backend_id)
        if self.backend_id == "":
            raise ValueError("backend_id must be nonempty when specified")
        object.__setattr__(
            self,
            "topology_summary",
            freeze_json_mapping(self.topology_summary),
        )
        object.__setattr__(
            self,
            "resume_metadata",
            freeze_json_mapping(self.resume_metadata),
        )
        object.__setattr__(
            self,
            "environment_summary",
            freeze_json_mapping(self.environment_summary),
        )
        object.__setattr__(
            self,
            "claims_not_made",
            _unique_strings(self.claims_not_made, "claims_not_made"),
        )
        object.__setattr__(self, "runtime_keys", runtime_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "global_step": self.global_step,
            "root_seed": self.root_seed,
            "runtime_keys": self.runtime_keys.to_dict(),
            "runtime_config": self.runtime_config.to_dict(),
            "environment_summary": json_value(self.environment_summary),
            "topology_summary": json_value(self.topology_summary),
            "precision_policy": self.precision_policy,
            "placement_policy": self.placement_policy,
            "backend_id": self.backend_id,
            "resume_metadata": json_value(self.resume_metadata),
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeState:
        return cls(
            schema_version=str(
                payload.get("schema_version", RUNTIME_STATE_SCHEMA_VERSION)
            ),
            runtime_id=str(payload["runtime_id"]),
            global_step=_required_int(payload["global_step"], "global_step"),
            root_seed=_required_int(payload["root_seed"], "root_seed"),
            runtime_keys=(
                None
                if payload.get("runtime_keys") is None
                else RuntimeKeys.from_dict(
                    _mapping(payload["runtime_keys"], "runtime_keys")
                )
            ),
            runtime_config=RuntimeConfig.from_dict(
                _mapping(payload["runtime_config"], "runtime_config")
            ),
            environment_summary=_mapping(
                payload.get("environment_summary", {}),
                "environment_summary",
            ),
            topology_summary=_mapping(
                payload.get("topology_summary", {}),
                "topology_summary",
            ),
            precision_policy=str(payload["precision_policy"]),
            placement_policy=str(payload["placement_policy"]),
            backend_id=_optional_string(payload.get("backend_id")),
            resume_metadata=_mapping(
                payload.get("resume_metadata", {}),
                "resume_metadata",
            ),
            claims_not_made=_string_tuple(
                payload.get(
                    "claims_not_made",
                    (
                        "model_parameters_not_included",
                        "optimizer_state_not_included",
                        "compiled_executables_not_included",
                        "architecture_state_not_included",
                    ),
                ),
                "claims_not_made",
            ),
        )


def freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise TypeError("JSON mapping keys must be strings")
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_json_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"value is not finite JSON data: {type(value).__name__}")


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    result = _string_tuple(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _unique_sorted_strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(sorted(_unique_strings(value, name)))


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{name} must contain integers")
    return result


def _unique_nonnegative_integers(value: Any, name: str) -> tuple[int, ...]:
    result = _integer_tuple(value, name)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else _required_int(value, "optional integer")


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else _required_bool(value, "optional boolean")


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_string(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _require_optional_string(name: str, value: Any) -> None:
    if value is not None:
        _require_string(name, value)


def _require_integer(name: str, value: Any) -> None:
    _required_int(value, name)


def _require_optional_integer(name: str, value: Any) -> None:
    if value is not None:
        _require_integer(name, value)


def _require_boolean(name: str, value: Any) -> None:
    _required_bool(value, name)


def _require_optional_boolean(name: str, value: Any) -> None:
    if value is not None:
        _require_boolean(name, value)
