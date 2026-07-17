"""Explicit registry for complete objective-plugin identities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from radjax_student.contracts import (
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
    canonical_objective_json,
)
from radjax_student.objectives.protocols import (
    JaxObjectiveExecution,
    ObjectivePlugin,
)

# This marker is deliberately process-local and never serialized.  It proves
# that a selection was issued by a registry without making a registry object or
# executable callable part of lifecycle identity.
_REGISTRY_SELECTION_MARKER = object()


def implementation_identity_for(
    plugin: ObjectivePlugin, profile: ObjectiveCapabilityProfile
) -> str:
    """Derive portable implementation identity without source bytes or addresses."""

    payload = {
        "module": type(plugin).__module__,
        "qualname": type(plugin).__qualname__,
        "identity": plugin.objective_identity().to_dict(),
        "capability_profile_digest": profile.digest,
        "execution_contract_version": plugin.execution_contract_version(),
    }
    digest = hashlib.sha256(canonical_objective_json(payload)).hexdigest()[:24]
    return f"radjax.objective_impl.impl_{digest}"


@dataclass(frozen=True)
class ObjectiveRegistrySelection:
    """A nonserializable registry selection paired with stable inspection data."""

    identity: ObjectiveIdentity
    profile: ObjectiveCapabilityProfile
    implementation_identity: str
    registry_identity: str
    plugin: ObjectivePlugin = field(repr=False, compare=False)
    _registration_marker: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ObjectiveIdentity):
            raise TypeError("objective registry selection requires ObjectiveIdentity")
        if not isinstance(self.profile, ObjectiveCapabilityProfile):
            raise TypeError("objective registry selection requires capability profile")
        if self.profile.identity != self.identity:
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "registry selection profile identity does not match objective identity",
            )
        if not isinstance(self.plugin, ObjectivePlugin):
            raise ObjectiveContractError(
                "objective_plugin_invalid",
                "registry selection requires complete plugin",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "objective_id": self.identity.objective_id,
            "objective_version": self.identity.objective_version,
            "capability_profile_digest": self.profile.digest,
            "implementation_identity": self.implementation_identity,
            "registry_identity": self.registry_identity,
        }

    @property
    def is_registry_selected(self) -> bool:
        """Whether this selection was issued by the public registry boundary."""

        return self._registration_marker is _REGISTRY_SELECTION_MARKER


@dataclass
class ObjectiveRegistry:
    """Manual objective selection; no discovery or fallback behavior exists."""

    registry_id: str = "radjax.objectives.registry.v1"
    _plugins: dict[ObjectiveIdentity, ObjectiveRegistrySelection] = field(
        default_factory=dict
    )

    def register(self, plugin: ObjectivePlugin) -> ObjectiveRegistrySelection:
        if not isinstance(plugin, ObjectivePlugin):
            raise ObjectiveContractError(
                "objective_plugin_invalid",
                "objective registry requires complete ObjectivePlugin implementations",
            )
        identity = plugin.objective_identity()
        if not isinstance(identity, ObjectiveIdentity):
            raise ObjectiveContractError(
                "objective_plugin_invalid",
                "objective plugin must return ObjectiveIdentity",
            )
        if (plugin.objective_id, plugin.objective_version) != (
            identity.objective_id,
            identity.objective_version,
        ):
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "objective plugin attributes must match objective identity",
            )
        profile = plugin.capability_profile()
        if not isinstance(profile, ObjectiveCapabilityProfile):
            raise ObjectiveContractError(
                "objective_capability_missing",
                "objective plugin must return ObjectiveCapabilityProfile",
            )
        if profile.identity != identity:
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "objective profile identity does not match plugin identity",
            )
        declares_jax = profile.supports("objective.jax_execution_v1")
        implements_jax = isinstance(plugin, JaxObjectiveExecution)
        if declares_jax != implements_jax:
            raise ObjectiveContractError(
                "objective_capability_missing",
                "JAX objective declaration and implementation must agree",
            )
        implementation_identity = implementation_identity_for(plugin, profile)
        if identity in self._plugins:
            existing = self._plugins[identity]
            code = (
                "objective_implementation_identity_mismatch"
                if existing.implementation_identity != implementation_identity
                else "objective_plugin_invalid"
            )
            raise ObjectiveContractError(
                code,
                "an objective identity is already registered",
                details={"identity": identity.to_dict()},
            )
        selection = ObjectiveRegistrySelection(
            identity,
            profile,
            implementation_identity,
            self.registry_id,
            plugin,
            _REGISTRY_SELECTION_MARKER,
        )
        self._plugins[identity] = selection
        return selection

    def select(self, identity: ObjectiveIdentity) -> ObjectiveRegistrySelection:
        if not isinstance(identity, ObjectiveIdentity):
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "objective registry requires ObjectiveIdentity",
            )
        try:
            return self._plugins[identity]
        except KeyError as error:
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "objective identity is not registered",
                details={"identity": identity.to_dict()},
            ) from error

    def execution_descriptor(
        self,
        *,
        selection: ObjectiveRegistrySelection,
        config: ObjectiveConfig,
        resolved_selection: ResolvedObjectiveSelection,
    ) -> ObjectiveExecutionDescriptor:
        if selection.registry_identity != self.registry_id:
            raise ObjectiveContractError(
                "objective_identity_mismatch", "selection belongs to another registry"
            )
        if not selection.is_registry_selected:
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "objective selection was not issued by ObjectiveRegistry",
            )
        stored = self.select(selection.identity)
        if stored.implementation_identity != selection.implementation_identity:
            raise ObjectiveContractError(
                "objective_implementation_identity_mismatch",
                "selected objective implementation is not the registry implementation",
            )
        if (
            not isinstance(config, ObjectiveConfig)
            or config.identity != selection.identity
        ):
            raise ObjectiveContractError(
                "objective_config_identity_mismatch",
                "objective config identity does not match selected plugin",
            )
        selection.plugin.validate_config(config)
        selection.plugin.validate_resolved_surface(resolved_selection)
        return ObjectiveExecutionDescriptor(
            identity=selection.identity,
            capability_profile_digest=selection.profile.digest,
            config_digest=config.digest,
            resolved_surface_identity=resolved_selection.digest,
            metric_schema_id=selection.profile.metric_schema_id,
            implementation_identity=selection.implementation_identity,
        )

    def inspect(self) -> tuple[Mapping[str, str], ...]:
        return tuple(
            MappingProxyType(item.to_dict())
            for _, item in sorted(
                self._plugins.items(),
                key=lambda entry: (
                    entry[0].objective_id,
                    entry[0].objective_version,
                ),
            )
        )


__all__ = [
    "ObjectiveRegistry",
    "ObjectiveRegistrySelection",
    "implementation_identity_for",
]
