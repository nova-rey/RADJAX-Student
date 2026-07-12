"""Declarative runtime backend registration without backend initialization."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from radjax_student.runtime.errors import RuntimeContractError, RuntimeIssue
from radjax_student.runtime.inspection import (
    RuntimeInspection,
    inspect_runtime_environment,
)
from radjax_student.runtime.models import (
    CompilationOptions,
    ExecutionContext,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
)

BackendAvailabilityStatus = Literal["available", "unavailable"]
SUPPORTED_RUNTIME_PLATFORMS: tuple[str, ...] = ("cpu", "gpu", "tpu", "metal")


@dataclass(frozen=True)
class RuntimeBackendAvailability:
    """Observed availability, separate from registration and declared support."""

    status: BackendAvailabilityStatus
    reasons: tuple[RuntimeIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("available", "unavailable"):
            raise ValueError(
                "backend availability status must be available or unavailable"
            )
        reasons = tuple(self.reasons)
        if any(not isinstance(item, RuntimeIssue) for item in reasons):
            raise TypeError("availability reasons must contain RuntimeIssue values")
        if self.status == "available" and reasons:
            raise ValueError("available backend cannot have unavailability reasons")
        if self.status == "unavailable" and not reasons:
            raise ValueError("unavailable backend must explain why")
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": [item.to_dict() for item in self.reasons],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeBackendAvailability:
        raw_reasons = payload.get("reasons", ())
        if not isinstance(raw_reasons, (list, tuple)):
            raise TypeError("availability reasons must be a list or tuple")
        return cls(
            status=str(payload["status"]),
            reasons=tuple(
                RuntimeIssue.from_dict(_mapping(item, "availability reason"))
                for item in raw_reasons
            ),
        )


@dataclass(frozen=True)
class RuntimeBackendDescriptor:
    """Serializable backend declaration with no implementation object attached."""

    backend_id: str
    implementation_version: str
    supported_platforms: tuple[str, ...]
    capability_profile: RuntimeCapabilityProfile
    availability: RuntimeBackendAvailability
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id:
            raise ValueError("backend_id must be a nonempty string")
        if (
            not isinstance(self.implementation_version, str)
            or not self.implementation_version
        ):
            raise ValueError("implementation_version must be a nonempty string")
        platforms = tuple(sorted(set(self.supported_platforms)))
        if not platforms or any(
            not isinstance(item, str) or not item for item in platforms
        ):
            raise ValueError("supported_platforms must contain nonempty strings")
        if not isinstance(self.capability_profile, RuntimeCapabilityProfile):
            raise TypeError("capability_profile must be RuntimeCapabilityProfile")
        if self.capability_profile.backend_id != self.backend_id:
            raise ValueError("descriptor and capability profile backend IDs must match")
        if not isinstance(self.availability, RuntimeBackendAvailability):
            raise TypeError("availability must be RuntimeBackendAvailability")
        notes = tuple(self.notes)
        if any(not isinstance(item, str) or not item for item in notes):
            raise ValueError("notes must contain nonempty strings")
        object.__setattr__(self, "supported_platforms", platforms)
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "implementation_version": self.implementation_version,
            "supported_platforms": list(self.supported_platforms),
            "capability_profile": self.capability_profile.to_dict(),
            "availability": self.availability.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeBackendDescriptor:
        platforms = payload.get("supported_platforms", ())
        notes = payload.get("notes", ())
        if not isinstance(platforms, (list, tuple)):
            raise TypeError("supported_platforms must be a list or tuple")
        if not isinstance(notes, (list, tuple)):
            raise TypeError("notes must be a list or tuple")
        return cls(
            backend_id=str(payload["backend_id"]),
            implementation_version=str(payload["implementation_version"]),
            supported_platforms=tuple(str(item) for item in platforms),
            capability_profile=RuntimeCapabilityProfile.from_dict(
                _mapping(payload["capability_profile"], "capability_profile")
            ),
            availability=RuntimeBackendAvailability.from_dict(
                _mapping(payload["availability"], "availability")
            ),
            notes=tuple(str(item) for item in notes),
        )


class DeclarativeRuntimeBackend(Protocol):
    """The selection-only subset of a backend declaration."""

    @property
    def backend_id(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def supported_platforms(self) -> tuple[str, ...]: ...

    @property
    def notes(self) -> tuple[str, ...]: ...

    def capability_profile(self) -> RuntimeCapabilityProfile: ...

    def availability(
        self, inspection: RuntimeInspection
    ) -> RuntimeBackendAvailability: ...


class RuntimeBackendRegistry:
    """Explicit, instance-owned registry with deterministic presentation order."""

    def __init__(self) -> None:
        self._backends: dict[str, DeclarativeRuntimeBackend] = {}

    def register(
        self,
        backend: DeclarativeRuntimeBackend,
        *,
        replace: bool = False,
    ) -> None:
        backend_id = _backend_id(backend)
        if backend_id in self._backends and not replace:
            raise RuntimeContractError(
                "runtime_backend_duplicate",
                "a runtime backend is already registered with this ID",
                details={"backend_id": backend_id},
            )
        self._backends[backend_id] = backend

    def unregister(self, backend_id: str) -> DeclarativeRuntimeBackend:
        try:
            return self._backends.pop(backend_id)
        except KeyError as exc:
            raise RuntimeContractError(
                "runtime_backend_not_found",
                "cannot unregister a backend that is not registered",
                details={"backend_id": backend_id},
            ) from exc

    def get(self, backend_id: str) -> DeclarativeRuntimeBackend | None:
        return self._backends.get(backend_id)

    def contains(self, backend_id: str) -> bool:
        return backend_id in self._backends

    def list_backends(self) -> tuple[DeclarativeRuntimeBackend, ...]:
        return tuple(self._backends[item] for item in sorted(self._backends))

    def describe(
        self, inspection: RuntimeInspection
    ) -> tuple[RuntimeBackendDescriptor, ...]:
        """Evaluate declarations against already observed facts only."""

        return tuple(
            _descriptor(backend, inspection) for backend in self.list_backends()
        )


class JaxRuntimeBackend:
    """JAX runtime declaration with CPU smoke and selected-device portability ops."""

    backend_id = "jax"
    implementation_version = "p2.9"
    supported_platforms = SUPPORTED_RUNTIME_PLATFORMS
    notes = (
        "P2.9 adds one shared selected-device portability smoke path.",
        "Declared capabilities beyond tested paths are not execution proof.",
    )

    def __init__(self) -> None:
        self._cpu_contexts: dict[str, tuple[Any, Any]] = {}

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="jax.runtime.p2.9",
            backend_id=self.backend_id,
            version=1,
            capabilities=(
                "compilation.jit_v1",
                "execution.argument_donation_v1",
                "execution.eager_v1",
                "execution.static_arguments_v1",
                "execution.synchronize_v1",
                "placement.single_device_v1",
                "runtime.single_process_v1",
                "state.runtime_envelope_v1",
            ),
            non_capabilities=(
                "placement.data_sharded_v1",
                "placement.model_sharded_v1",
                "placement.replicated_v1",
                "runtime.multi_process_v1",
            ),
            notes=("P2.9 proves only selected single-device portability smoke paths.",),
        )

    def availability(self, inspection: RuntimeInspection) -> RuntimeBackendAvailability:
        if inspection.environment.jax_available:
            return RuntimeBackendAvailability("available")
        return RuntimeBackendAvailability(
            "unavailable",
            reasons=(
                RuntimeIssue.create(
                    "runtime_backend_unavailable",
                    "JAX is not available in the observed environment",
                    backend_id=self.backend_id,
                    inspection_warning_codes=inspection.environment.warnings,
                ),
            ),
        )

    def inspect_environment(self) -> RuntimeEnvironment:
        return inspect_runtime_environment().environment

    def initialize(self, config: RuntimeConfig) -> ExecutionContext:
        del config
        raise _execution_requires_selection_error()

    def place(self, value: Any, placement: str) -> Any:
        del value, placement
        raise _execution_requires_selection_error()

    def compile(
        self,
        function: Callable[..., Any],
        options: CompilationOptions,
    ) -> Callable[..., Any]:
        del function, options
        raise _execution_requires_selection_error()

    def synchronize(self, value: Any) -> Any:
        del value
        raise _execution_requires_selection_error()

    def close(self, context: ExecutionContext) -> None:
        del context
        raise _execution_requires_selection_error()

    def initialize_cpu_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device_descriptor: Any,
    ) -> ExecutionContext:
        """Initialize one selected CPU context after P2.3 has approved it."""

        jax_module = _import_jax()
        device = _selected_jax_cpu_device(jax_module, device_descriptor)
        runtime_id = f"jax-cpu-smoke-seed-{config.seed}"
        context = ExecutionContext(
            backend_id=self.backend_id,
            environment=inspection.environment,
            device_inventory=inspection.device_inventory,
            capabilities=selection.selected_backend.capability_profile,
            root_seed=config.seed,
            runtime_id=runtime_id,
            metadata={
                "selected_platform": "cpu",
                "selected_device_id": device_descriptor.device_id,
                "placement_policy": config.placement_policy,
            },
        )
        self._cpu_contexts[runtime_id] = (jax_module, device)
        return context

    def place_cpu_value(self, context: ExecutionContext, value: Any) -> Any:
        jax_module, device = self._cpu_context(context)
        array_module = getattr(jax_module, "numpy", None)
        normalized = value if array_module is None else array_module.asarray(value)
        return jax_module.device_put(normalized, device)

    def execute_cpu_smoke(self, context: ExecutionContext, value: Any) -> Any:
        del context
        return value * 2 + 1

    def synchronize_cpu_value(self, context: ExecutionContext, value: Any) -> Any:
        del context
        return value.block_until_ready()

    def close_cpu_context(self, context: ExecutionContext) -> None:
        self._cpu_contexts.pop(context.runtime_id, None)

    def initialize_portability_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device_descriptor: Any,
    ) -> ExecutionContext:
        """Initialize one selected local CPU, GPU, or TPU device for P2.9."""

        platform = selection.selected_platform
        if platform not in ("cpu", "gpu", "tpu"):
            raise RuntimeContractError(
                "runtime_device_selection_failed",
                "portability execution requires one selected CPU, GPU, or TPU",
                details={"selected_platform": platform},
            )
        jax_module = _import_jax()
        device = _selected_jax_device(jax_module, device_descriptor, platform)
        runtime_id = f"jax-{platform}-portability-seed-{config.seed}"
        context = ExecutionContext(
            backend_id=self.backend_id,
            environment=inspection.environment,
            device_inventory=inspection.device_inventory,
            capabilities=selection.selected_backend.capability_profile,
            root_seed=config.seed,
            runtime_id=runtime_id,
            metadata={
                "selected_platform": platform,
                "selected_device_id": device_descriptor.device_id,
                "placement_policy": config.placement_policy,
                "portability_smoke": True,
            },
        )
        self._cpu_contexts[runtime_id] = (jax_module, device)
        return context

    def place_portability_value(self, context: ExecutionContext, value: Any) -> Any:
        """Materialize and explicitly place the shared P2.9 input on one device."""

        jax_module, device = self._cpu_context(context)
        return jax_module.device_put(jax_module.numpy.asarray(value), device)

    def place_execution_pytree(
        self, context: ExecutionContext, value: Any, *, precision_policy: str
    ) -> Any:
        """Place a complete JAX input pytree on the selected runtime device."""

        if precision_policy not in {
            "float32",
            "bfloat16",
            "float16",
            "mixed",
            "automatic",
            "unspecified",
        }:
            raise RuntimeContractError(
                "runtime_placement_failed", "unsupported execution precision policy"
            )
        jax_module, device = self._cpu_context(context)
        array_module = jax_module.numpy
        dtype = {
            "float32": array_module.float32,
            "bfloat16": array_module.bfloat16,
            "float16": array_module.float16,
        }.get(precision_policy)

        def place_leaf(leaf: Any) -> Any:
            array = array_module.asarray(leaf)
            if dtype is not None and getattr(array.dtype, "kind", "") == "f":
                array = array.astype(dtype)
            return jax_module.device_put(array, device)

        return jax_module.tree_util.tree_map(place_leaf, value)

    def close_portability_context(self, context: ExecutionContext) -> None:
        self._cpu_contexts.pop(context.runtime_id, None)

    def prepare_runtime_execution(
        self,
        context: ExecutionContext,
        function: Callable[..., Any],
        request: Any,
        mode: str,
    ) -> _JaxExecutionHandle:
        del context
        jax_module = _import_jax()
        options = request.compilation_options
        if mode == "eager":
            return _JaxExecutionHandle(
                mode=mode,
                jax_module=jax_module,
                callable=function,
                cache_policy=options.cache_policy,
            )
        jit_kwargs: dict[str, Any] = {}
        if options.static_arg_names:
            jit_kwargs["static_argnames"] = options.static_arg_names
        if options.static_arg_positions:
            jit_kwargs["static_argnums"] = options.static_arg_positions
        if options.donate_arg_names:
            jit_kwargs["donate_argnames"] = options.donate_arg_names
        if options.donate_arg_positions:
            jit_kwargs["donate_argnums"] = options.donate_arg_positions
        return _JaxExecutionHandle(
            mode=mode,
            jax_module=jax_module,
            callable=jax_module.jit(function, **jit_kwargs),
            cache_policy=options.cache_policy,
        )

    def compile_runtime_execution(
        self,
        context: ExecutionContext,
        handle: _JaxExecutionHandle,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[_JaxExecutionHandle, bool]:
        del context
        if handle.mode == "eager":
            return handle, False
        if handle.executable is not None and handle.cache_policy == "reuse":
            return handle, False
        handle.executable = handle.callable.lower(*args, **kwargs).compile()
        return handle, True

    def dispatch_runtime_execution(
        self,
        context: ExecutionContext,
        handle: _JaxExecutionHandle,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        del context
        executable = (
            handle.executable if handle.executable is not None else handle.callable
        )
        return executable(*args, **kwargs)

    def synchronize_runtime_execution(
        self,
        context: ExecutionContext,
        output: Any,
    ) -> Any:
        del context
        return self._jax_for_execution(output).block_until_ready(output)

    def _cpu_context(self, context: ExecutionContext) -> tuple[Any, Any]:
        try:
            return self._cpu_contexts[context.runtime_id]
        except KeyError as exc:
            raise RuntimeContractError(
                "runtime_backend_initialization_failed",
                "JAX CPU context is not active",
                details={"runtime_id": context.runtime_id},
            ) from exc

    def _jax_for_execution(self, output: Any) -> Any:
        del output
        if self._cpu_contexts:
            return next(iter(self._cpu_contexts.values()))[0]
        return _import_jax()


@dataclass
class _JaxExecutionHandle:
    mode: str
    jax_module: Any
    callable: Callable[..., Any]
    cache_policy: str
    executable: Callable[..., Any] | None = None


@dataclass(frozen=True)
class FakeRuntimeBackend:
    """Small declaration-only backend intended for deterministic selection tests."""

    backend_id: str = "fake"
    implementation_version: str = "test-v1"
    supported_platforms: tuple[str, ...] = ("cpu",)
    capabilities: tuple[str, ...] = ("runtime.single_process_v1",)
    non_capabilities: tuple[str, ...] = ()
    available: bool = True
    notes: tuple[str, ...] = ("Test-only declaration; excluded from defaults.",)

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id=f"{self.backend_id}.runtime.test",
            backend_id=self.backend_id,
            version=1,
            capabilities=self.capabilities,
            non_capabilities=self.non_capabilities,
            notes=self.notes,
        )

    def availability(self, inspection: RuntimeInspection) -> RuntimeBackendAvailability:
        del inspection
        if self.available:
            return RuntimeBackendAvailability("available")
        return RuntimeBackendAvailability(
            "unavailable",
            reasons=(
                RuntimeIssue.create(
                    "runtime_backend_unavailable",
                    "test backend was declared unavailable",
                    backend_id=self.backend_id,
                ),
            ),
        )


def build_default_runtime_registry() -> RuntimeBackendRegistry:
    """Return a fresh deterministic registry without importing or inspecting JAX."""

    registry = RuntimeBackendRegistry()
    registry.register(JaxRuntimeBackend())
    return registry


def _descriptor(
    backend: DeclarativeRuntimeBackend,
    inspection: RuntimeInspection,
) -> RuntimeBackendDescriptor:
    backend_id = _backend_id(backend)
    implementation_version = getattr(backend, "implementation_version", None)
    platforms = getattr(backend, "supported_platforms", None)
    notes = getattr(backend, "notes", ())
    if not isinstance(implementation_version, str) or not implementation_version:
        raise TypeError(
            "runtime backend implementation_version must be a nonempty string"
        )
    if not isinstance(platforms, tuple):
        raise TypeError("runtime backend supported_platforms must be a tuple")
    if not isinstance(notes, tuple):
        raise TypeError("runtime backend notes must be a tuple")
    return RuntimeBackendDescriptor(
        backend_id=backend_id,
        implementation_version=implementation_version,
        supported_platforms=platforms,
        capability_profile=backend.capability_profile(),
        availability=backend.availability(inspection),
        notes=notes,
    )


def _backend_id(backend: DeclarativeRuntimeBackend) -> str:
    backend_id = getattr(backend, "backend_id", None)
    if not isinstance(backend_id, str) or not backend_id:
        raise TypeError("runtime backend backend_id must be a nonempty string")
    return backend_id


def _execution_requires_selection_error() -> RuntimeContractError:
    return RuntimeContractError(
        "runtime_initialization_failed",
        "JAX execution requires the P2.4 selected CPU smoke path",
    )


def _import_jax() -> Any:
    try:
        return importlib.import_module("jax")
    except Exception as exc:
        raise RuntimeContractError(
            "runtime_backend_initialization_failed",
            "JAX could not be imported for the selected CPU smoke",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        ) from exc


def _selected_jax_cpu_device(jax_module: Any, descriptor: Any) -> Any:
    return _selected_jax_device(jax_module, descriptor, "cpu")


def _selected_jax_device(jax_module: Any, descriptor: Any, platform: str) -> Any:
    try:
        raw_devices = tuple(jax_module.devices(platform))
    except Exception as exc:
        raise RuntimeContractError(
            "runtime_device_selection_failed",
            "JAX could not list devices for the selected portability smoke",
            details={
                "platform": platform,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        ) from exc
    reported_id = descriptor.metadata.get("jax_reported_device_id")
    matches = [
        device
        for device in raw_devices
        if getattr(device, "platform", None) == platform
        and getattr(device, "id", None) == reported_id
        and getattr(device, "process_index", None) == descriptor.process_index
    ]
    if not matches:
        raise RuntimeContractError(
            "runtime_device_selection_failed",
            "selected inspected device is unavailable to JAX execution",
            details={"platform": platform, "device_id": descriptor.device_id},
        )
    return matches[0]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value
