"""Complete objective-plugin protocols; these interfaces do not import JAX."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from radjax_student.contracts import (
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
)


@runtime_checkable
class ObjectivePlugin(Protocol):
    """The sole production identity for objective semantics and metrics."""

    objective_id: str
    objective_version: str

    def objective_identity(self) -> ObjectiveIdentity: ...

    def capability_profile(self) -> ObjectiveCapabilityProfile: ...

    def validate_config(self, config: ObjectiveConfig) -> None: ...

    def validate_resolved_surface(
        self, selection: ResolvedObjectiveSelection
    ) -> None: ...

    def validate_targets(self, targets: Any) -> None: ...

    def validate_metrics(self, metrics: Mapping[str, Any]) -> None: ...

    def execution_contract_version(self) -> str: ...


@runtime_checkable
class JaxObjectiveExecution(Protocol):
    """Optional JAX execution capability of the same complete objective plugin."""

    def evaluate_jax(
        self,
        *,
        surface: Any,
        targets: Any,
        weights: Any,
        config: ObjectiveConfig,
    ) -> tuple[Any, Mapping[str, Any]]: ...


@runtime_checkable
class JaxObjectivePlugin(ObjectivePlugin, JaxObjectiveExecution, Protocol):
    """The only production JAX objective contract."""


__all__ = ["JaxObjectiveExecution", "JaxObjectivePlugin", "ObjectivePlugin"]
