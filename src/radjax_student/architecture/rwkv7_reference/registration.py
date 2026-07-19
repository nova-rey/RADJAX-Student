"""Explicit caller-owned registration for the RWKV-7 reference plugin."""

from __future__ import annotations

from radjax_student.architecture.registry import ArchitectureRegistry
from radjax_student.architecture.rwkv7_reference.plugin import RWKV7ReferencePlugin


def register_rwkv7_reference(registry: ArchitectureRegistry) -> RWKV7ReferencePlugin:
    """Register one static plugin; default registration is intentionally absent."""

    if not isinstance(registry, ArchitectureRegistry):
        raise TypeError("registry must be ArchitectureRegistry")
    plugin = RWKV7ReferencePlugin()
    registry.register(plugin)
    return plugin


__all__ = ["register_rwkv7_reference"]
