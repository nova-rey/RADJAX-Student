"""Explicit deterministic optimizer registration."""

from __future__ import annotations

from dataclasses import dataclass, field

from radjax_student.optimizers.errors import OptimizerContractError
from radjax_student.optimizers.protocols import OptimizerBackend


@dataclass
class OptimizerRegistry:
    _backends: dict[str, OptimizerBackend] = field(default_factory=dict)

    def register(self, backend: OptimizerBackend) -> None:
        if not isinstance(backend.optimizer_id, str) or not backend.optimizer_id:
            raise OptimizerContractError(
                "optimizer_config_invalid", "optimizer ID must be a nonempty string"
            )
        if backend.optimizer_id in self._backends:
            raise OptimizerContractError(
                "optimizer_backend_duplicate",
                "optimizer ID is already registered",
                details={"optimizer_id": backend.optimizer_id},
            )
        self._backends[backend.optimizer_id] = backend

    def get(self, optimizer_id: str) -> OptimizerBackend:
        try:
            return self._backends[optimizer_id]
        except KeyError as exc:
            raise OptimizerContractError(
                "optimizer_backend_not_found",
                "optimizer backend is not registered",
                details={"optimizer_id": optimizer_id},
            ) from exc

    def list_optimizers(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))
