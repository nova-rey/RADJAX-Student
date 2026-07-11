"""Passive future seams; P3.1 declares contracts without invoking learning."""

from __future__ import annotations

from typing import Protocol

from radjax_student.learning.models import LearningBatch, LossResult
from radjax_student.learning.scopes import (
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


class ObjectiveEvaluator(Protocol):
    """A future architecture or behavior adapter that evaluates a generic objective."""

    def evaluate(self, batch: LearningBatch, scope: ObjectiveScope) -> LossResult: ...


class UpdateScopeResolver(Protocol):
    """A future architecture adapter that resolves generic update intent."""

    def resolve_update_scope(self, scope: UpdateScope) -> ResolvedUpdateSelection: ...
