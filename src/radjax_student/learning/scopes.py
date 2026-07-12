"""Compatibility re-exports for shared scope contracts."""

from radjax_student.contracts.scopes import (
    OBJECTIVE_SCOPE_KINDS,
    UPDATE_SCOPE_KINDS,
    ObjectiveScope,
    ObjectiveScopeKind,
    ResolvedUpdateSelection,
    UpdateScope,
    UpdateScopeKind,
)

__all__ = [
    "OBJECTIVE_SCOPE_KINDS",
    "UPDATE_SCOPE_KINDS",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "ResolvedUpdateSelection",
    "UpdateScope",
    "UpdateScopeKind",
]
