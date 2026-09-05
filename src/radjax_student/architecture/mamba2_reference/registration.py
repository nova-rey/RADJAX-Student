"""Explicit caller-owned registration for the Mamba-2 reference plugin."""

from __future__ import annotations

from radjax_student.architecture.mamba2_reference.plugin import Mamba2ReferencePlugin
from radjax_student.architecture.registry import ArchitectureRegistry


def register_mamba2_reference(registry: ArchitectureRegistry) -> Mamba2ReferencePlugin:
    if not isinstance(registry, ArchitectureRegistry):
        raise TypeError("registry must be ArchitectureRegistry")
    plugin = Mamba2ReferencePlugin()
    registry.register(plugin)
    return plugin


__all__ = ["register_mamba2_reference"]
