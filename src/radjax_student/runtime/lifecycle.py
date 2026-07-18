"""Runtime-owned context and key binding for learning lifecycle assembly.

This module deliberately owns only runtime facts: registry selection, the
selected execution context, and a named key stream derived from that context.
It does not know about architectures, objectives, optimizers, or learning
lifecycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radjax_student.runtime.inspection import inspect_runtime_environment
from radjax_student.runtime.keys import RuntimeKeyStream
from radjax_student.runtime.models import ExecutionContext, RuntimeConfig
from radjax_student.runtime.registry import RuntimeBackendRegistry
from radjax_student.runtime.selection import (
    RuntimeSelectionResult,
    select_runtime_backend,
)


@dataclass(frozen=True)
class RuntimeLifecycleBinding:
    """The exact runtime-owned values required by a learning lifecycle."""

    backend: Any
    context: ExecutionContext
    key_stream: RuntimeKeyStream
    selection: RuntimeSelectionResult

    def __post_init__(self) -> None:
        if not isinstance(self.context, ExecutionContext):
            raise TypeError("runtime lifecycle binding requires ExecutionContext")
        if not isinstance(self.key_stream, RuntimeKeyStream):
            raise TypeError("runtime lifecycle binding requires RuntimeKeyStream")
        if not isinstance(self.selection, RuntimeSelectionResult):
            raise TypeError("runtime lifecycle binding requires RuntimeSelectionResult")
        if not self.selection.ok or self.selection.selected_backend is None:
            raise ValueError("runtime lifecycle binding requires a selected backend")


def bind_runtime_for_learning(
    config: RuntimeConfig,
    *,
    implementation_version: str,
    root_seed: int,
    rng_slot: str,
    registry: RuntimeBackendRegistry,
) -> RuntimeLifecycleBinding:
    """Select, initialize, and bind runtime state through runtime authority.

    The learning assembler supplies declarative request values only.  Device
    lookup, context construction, and key-stream derivation all remain here so
    no learning caller can select a raw device or fabricate a free-standing
    stream.
    """

    if not isinstance(config, RuntimeConfig):
        raise TypeError("runtime config must be RuntimeConfig")
    if not isinstance(registry, RuntimeBackendRegistry):
        raise TypeError("runtime registry must be RuntimeBackendRegistry")
    if not isinstance(implementation_version, str) or not implementation_version:
        raise ValueError("runtime implementation version is required")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("runtime root seed must be a nonnegative integer")
    if not isinstance(rng_slot, str) or not rng_slot:
        raise ValueError("runtime key slot is required")
    if config.seed != root_seed:
        raise ValueError("runtime config seed must match the requested root seed")

    backend = registry.get(config.backend_id)
    if backend is None:
        raise LookupError("requested runtime backend is not registered")
    if getattr(backend, "implementation_version", None) != implementation_version:
        raise ValueError("runtime implementation version does not match backend")

    inspection = inspect_runtime_environment()
    selection = select_runtime_backend(config, inspection, registry)
    if (
        not selection.ok
        or selection.selected_backend is None
        or selection.selected_platform is None
    ):
        raise ValueError("runtime selection did not produce one executable backend")
    if selection.selected_backend.backend_id != config.backend_id:
        raise ValueError("runtime selection chose a different backend")
    selected_device = next(
        (
            device
            for device in inspection.device_inventory.devices
            if device.platform == selection.selected_platform
        ),
        None,
    )
    if selected_device is None:
        raise ValueError("runtime selection has no selected device")
    initialize = getattr(backend, "initialize_portability_context", None)
    if not callable(initialize):
        raise TypeError("runtime backend cannot initialize a portability context")
    context = initialize(config, inspection, selection, selected_device)
    if not isinstance(context, ExecutionContext):
        raise TypeError("runtime backend did not return ExecutionContext")
    if context.runtime_keys is None:
        raise ValueError("runtime context did not provide runtime keys")
    key_stream = context.runtime_keys.stream(rng_slot)
    return RuntimeLifecycleBinding(backend, context, key_stream, selection)


__all__ = ["RuntimeLifecycleBinding", "bind_runtime_for_learning"]
