"""Explicit deterministic optimizer registration."""

from __future__ import annotations

from dataclasses import dataclass, field

from radjax_student.optimizers.errors import OptimizerContractError
from radjax_student.optimizers.models import OptimizerCapabilityProfile
from radjax_student.optimizers.protocols import JaxOptimizerExecution, OptimizerBackend


@dataclass
class OptimizerRegistry:
    _backends: dict[str, OptimizerBackend] = field(default_factory=dict)

    def register(
        self, backend: OptimizerBackend, *, registry_id: str | None = None
    ) -> None:
        if not isinstance(backend, OptimizerBackend):
            raise OptimizerContractError(
                "optimizer_config_invalid", "registry requires full OptimizerBackend"
            )
        if not isinstance(backend.optimizer_id, str) or not backend.optimizer_id:
            raise OptimizerContractError(
                "optimizer_config_invalid", "optimizer ID must be a nonempty string"
            )
        if registry_id is not None and registry_id != backend.optimizer_id:
            raise OptimizerContractError(
                "optimizer_config_invalid",
                "explicit registry ID must match the optimizer ID",
                details={
                    "registry_id": registry_id,
                    "optimizer_id": backend.optimizer_id,
                },
            )
        if backend.optimizer_id in self._backends:
            raise OptimizerContractError(
                "optimizer_backend_duplicate",
                "optimizer ID is already registered",
                details={"optimizer_id": backend.optimizer_id},
            )
        capability = backend.capability_profile()
        if not isinstance(capability, OptimizerCapabilityProfile):
            raise OptimizerContractError(
                "optimizer_capability_missing",
                "optimizer must return OptimizerCapabilityProfile",
            )
        if (
            capability.optimizer_id != backend.optimizer_id
            or capability.version != backend.optimizer_version
        ):
            raise OptimizerContractError(
                "optimizer_capability_missing",
                "optimizer identity must match its capability profile",
            )
        declares_jax = "optimizer.jax_execution_v1" in capability.capabilities
        if declares_jax != isinstance(backend, JaxOptimizerExecution):
            raise OptimizerContractError(
                "optimizer_jax_capability_missing",
                "JAX optimizer declaration and implementation must agree",
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
