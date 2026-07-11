"""Passive future seams; P3.1 declares contracts without invoking learning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from radjax_student.learning.models import LearningBatch, LossResult
from radjax_student.learning.scopes import (
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


class ObjectiveEvaluator(Protocol):
    """A future architecture or behavior adapter that evaluates a generic objective."""

    def evaluate(self, batch: LearningBatch, scope: ObjectiveScope) -> LossResult: ...


@runtime_checkable
class ForwardObjectiveEvaluator(Protocol):
    """Objective seam that consumes architecture output, never parameters."""

    def evaluate(
        self,
        surface: Any,
        targets: Mapping[str, Any],
        weights: Mapping[str, Any],
        objective_config: Any,
    ) -> tuple[Any, Mapping[str, Any]]: ...


class UpdateScopeResolver(Protocol):
    """A future architecture adapter that resolves generic update intent."""

    def resolve_update_scope(self, scope: UpdateScope) -> ResolvedUpdateSelection: ...
