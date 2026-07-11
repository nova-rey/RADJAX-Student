"""Pure-Python SGD test backend; it is not a learning loop."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.learning import MetricRecord
from radjax_student.optimizers.errors import OptimizerContractError, OptimizerIssue
from radjax_student.optimizers.models import (
    OptimizerCapabilityProfile,
    OptimizerConfig,
    OptimizerInitRequest,
    OptimizerInitResult,
    OptimizerState,
    OptimizerStateDescriptor,
    OptimizerUpdateRequest,
    OptimizerUpdateResult,
    ParameterUpdate,
)

SGD_OPTIMIZER_ID = "sgd.v1"
SGD_CAPABILITIES: tuple[str, ...] = (
    "optimizer.apply_updates_v1",
    "optimizer.initialize_state_v1",
    "optimizer.scoped_updates_v1",
    "optimizer.state_serialization_v1",
)


@dataclass(frozen=True)
class SgdOptimizer:
    """Scalar-mapping test backend proving update mechanics and scope masking."""

    optimizer_id: str = SGD_OPTIMIZER_ID
    optimizer_version: int = 1

    def capability_profile(self) -> OptimizerCapabilityProfile:
        return OptimizerCapabilityProfile(
            self.optimizer_id,
            self.optimizer_version,
            SGD_CAPABILITIES,
            metadata={"test_backend": True},
        )

    def validate_config(self, config: OptimizerConfig) -> None:
        if config.optimizer_id != self.optimizer_id:
            raise OptimizerContractError(
                "optimizer_config_invalid",
                "configuration optimizer ID does not match SGD",
                details={
                    "expected": self.optimizer_id,
                    "received": config.optimizer_id,
                },
            )
        if config.momentum not in (None, 0.0):
            raise OptimizerContractError(
                "optimizer_capability_missing",
                "SGD test backend does not implement momentum",
            )
        if (
            config.gradient_clip_mode != "disabled"
            or config.weight_decay_mode != "disabled"
        ):
            raise OptimizerContractError(
                "optimizer_capability_missing",
                "SGD test backend declares clipping and weight decay as unavailable",
            )

    def initialize_state(self, request: OptimizerInitRequest) -> OptimizerInitResult:
        self.validate_config(request.config)
        paths = request.parameter_catalog.paths
        state = OptimizerState(
            optimizer_id=self.optimizer_id,
            parameter_paths=paths,
            state_structure={
                "per_parameter_steps": "integer",
                "step_counter": "integer",
            },
            backend_state={"per_parameter_steps": {path: 0 for path in paths}},
            metadata={"test_backend": True},
        )
        return OptimizerInitResult(
            state,
            state_metadata={
                "selected_paths": list(
                    request.resolved_update_selection.selected_parameter_paths
                )
            },
            warnings=(
                OptimizerIssue(
                    "optimizer_test_backend",
                    "SGD is a scalar-mapping test backend, not a training proof.",
                ),
            ),
        )

    def apply_updates(self, request: OptimizerUpdateRequest) -> OptimizerUpdateResult:
        self.validate_config(request.config)
        if request.optimizer_state.optimizer_id != self.optimizer_id:
            raise OptimizerContractError(
                "optimizer_state_invalid", "optimizer state does not belong to SGD"
            )
        parameters, gradients = (
            _scalar_mapping(request.parameters, "parameters"),
            _scalar_mapping(request.gradients.values, "gradients"),
        )
        expected = set(request.optimizer_state.parameter_paths)
        if set(parameters) != expected or set(gradients) != expected:
            raise OptimizerContractError(
                "optimizer_gradient_structure_invalid",
                "parameters and gradients must match optimizer state paths",
                details={"expected_paths": sorted(expected)},
            )
        selected = set(request.resolved_update_selection.selected_parameter_paths)
        if not selected <= expected:
            raise OptimizerContractError(
                "optimizer_update_scope_invalid",
                "selected paths are absent from optimizer state",
            )
        for path, gradient in gradients.items():
            if not math.isfinite(gradient):
                raise OptimizerContractError(
                    "optimizer_gradient_nonfinite",
                    "gradient must be finite",
                    details={"path": path},
                )
        learning_rate = float(
            request.schedule_values.get("learning_rate", request.config.learning_rate)
        )
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise OptimizerContractError(
                "optimizer_update_failed",
                "resolved learning rate must be finite and positive",
            )
        updated_parameters = dict(parameters)
        old_backend_state = request.optimizer_state.backend_state
        old_steps = (
            old_backend_state.get("per_parameter_steps", {})
            if isinstance(old_backend_state, Mapping)
            else {}
        )
        new_steps = dict(old_steps)
        updates: list[ParameterUpdate] = []
        for path in sorted(expected):
            if path in selected:
                delta = -learning_rate * gradients[path]
                updated_parameters[path] = parameters[path] + delta
                new_steps[path] = int(new_steps.get(path, 0)) + 1
                updates.append(ParameterUpdate(path, True, abs(delta)))
            else:
                updates.append(
                    ParameterUpdate(
                        path, False, 0.0, metadata={"reason": "excluded_by_scope"}
                    )
                )
        state = OptimizerState(
            optimizer_id=self.optimizer_id,
            parameter_paths=request.optimizer_state.parameter_paths,
            step=request.optimizer_state.step + 1,
            state_structure=request.optimizer_state.state_structure,
            backend_state={"per_parameter_steps": new_steps},
            metadata=request.optimizer_state.metadata,
        )
        changed = tuple(
            sorted(
                path for path in expected if path in selected and gradients[path] != 0.0
            )
        )
        unchanged = tuple(sorted(expected - set(changed)))
        return OptimizerUpdateResult(
            updated_optimizer_state=state,
            updated_parameters=updated_parameters,
            parameter_updates=tuple(updates),
            changed_parameter_paths=changed,
            unchanged_parameter_paths=unchanged,
            update_metadata={
                "learning_rate": learning_rate,
                "selected_paths": sorted(selected),
            },
            metrics=(
                MetricRecord("optimizer.learning_rate", learning_rate, state.step),
            ),
            warnings=(
                OptimizerIssue(
                    "optimizer_partial_update_not_training_proof",
                    "Scoped SGD update proves masking mechanics only.",
                ),
            ),
        )

    def describe_state(self, state: OptimizerState) -> OptimizerStateDescriptor:
        if state.optimizer_id != self.optimizer_id:
            raise OptimizerContractError(
                "optimizer_state_invalid", "optimizer state does not belong to SGD"
            )
        return OptimizerStateDescriptor(
            self.optimizer_id,
            state.step,
            state.parameter_paths,
            ("step_counter",),
            len(state.parameter_paths),
            metadata={"test_backend": True},
        )


def _scalar_mapping(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise OptimizerContractError(
            "optimizer_gradient_structure_invalid",
            f"{name} must be a mapping of scalar values",
        )
    result: dict[str, float] = {}
    for path, scalar in value.items():
        if (
            not isinstance(path, str)
            or isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
        ):
            raise OptimizerContractError(
                "optimizer_gradient_structure_invalid",
                f"{name} must map paths to numeric values",
            )
        result[path] = float(scalar)
    return result
