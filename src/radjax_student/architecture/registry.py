"""Explicit deterministic registration for architecture plugins."""

from __future__ import annotations

from dataclasses import dataclass, field

from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.protocols import ArchitecturePlugin


@dataclass
class ArchitectureRegistry:
    """A manual registry; discovery and selection policy are intentionally absent."""

    _plugins: dict[str, ArchitecturePlugin] = field(default_factory=dict)

    def register(self, plugin: ArchitecturePlugin) -> None:
        architecture_id = plugin.architecture_id
        if not isinstance(architecture_id, str) or not architecture_id:
            raise ArchitectureContractError(
                "architecture_config_invalid",
                "architecture plugin ID must be a nonempty string",
            )
        if architecture_id in self._plugins:
            raise ArchitectureContractError(
                "architecture_plugin_duplicate",
                "architecture plugin ID is already registered",
                details={"architecture_id": architecture_id},
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
