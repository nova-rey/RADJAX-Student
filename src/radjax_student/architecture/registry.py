"""Explicit deterministic registration for architecture plugins."""

from __future__ import annotations

from dataclasses import dataclass, field

from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.models import ArchitectureCapabilityProfile
from radjax_student.architecture.protocols import (
    ArchitecturePlugin,
    JaxArchitectureExecution,
)


@dataclass
class ArchitectureRegistry:
    """A manual registry; discovery and selection policy are intentionally absent."""

    _plugins: dict[str, ArchitecturePlugin] = field(default_factory=dict)

    def register(
        self, plugin: ArchitecturePlugin, *, registry_id: str | None = None
    ) -> None:
        if not isinstance(plugin, ArchitecturePlugin):
            raise ArchitectureContractError(
                "architecture_plugin_invalid",
                "registry accepts complete ArchitecturePlugin implementations only",
            )
        architecture_id = plugin.architecture_id
        if not isinstance(architecture_id, str) or not architecture_id:
            raise ArchitectureContractError(
                "architecture_config_invalid",
                "architecture plugin ID must be a nonempty string",
            )
        if registry_id is not None and registry_id != architecture_id:
            raise ArchitectureContractError(
                "architecture_config_invalid",
                "explicit registry ID must match the plugin architecture ID",
                details={
                    "registry_id": registry_id,
                    "architecture_id": architecture_id,
                },
            )
        if architecture_id in self._plugins:
            raise ArchitectureContractError(
                "architecture_plugin_duplicate",
                "architecture plugin ID is already registered",
                details={"architecture_id": architecture_id},
            )
        capability = plugin.capability_profile()
        if not isinstance(capability, ArchitectureCapabilityProfile):
            raise ArchitectureContractError(
                "architecture_capability_missing",
                "architecture plugin must return ArchitectureCapabilityProfile",
                details={"architecture_id": architecture_id},
            )
        if (
            capability.architecture_id != architecture_id
            or capability.version != plugin.architecture_version
        ):
            raise ArchitectureContractError(
                "architecture_capability_missing",
                "architecture plugin identity must match its capability profile",
                details={
                    "architecture_id": architecture_id,
                    "architecture_version": plugin.architecture_version,
                    "profile_architecture_id": capability.architecture_id,
                    "profile_version": capability.version,
                },
            )
        declares_jax = capability.supports("architecture.jax_execution_v1")
        implements_jax = isinstance(plugin, JaxArchitectureExecution)
        if declares_jax != implements_jax:
            raise ArchitectureContractError(
                "architecture_capability_missing",
                "JAX execution declaration and implementation must agree",
                details={
                    "architecture_id": architecture_id,
                    "declares_jax_execution": declares_jax,
                    "implements_jax_execution": implements_jax,
                },
            )
        self._plugins[architecture_id] = plugin

    def get(self, architecture_id: str) -> ArchitecturePlugin:
        try:
            return self._plugins[architecture_id]
        except KeyError as exc:
            raise ArchitectureContractError(
                "architecture_plugin_not_found",
                "architecture plugin is not registered",
                details={"architecture_id": architecture_id},
            ) from exc

    def list_plugins(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
