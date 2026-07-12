"""Pure-Python SGD test backend; it is not a learning loop."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from radjax_student.contracts import (
    JaxOptimizerStateDescriptor,
    MetricRecord,
    ParameterTreeLayout,
)
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
    "optimizer.jax_execution_v1",
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

    def jax_state_descriptor(
        self, parameter_layout: ParameterTreeLayout
    ) -> JaxOptimizerStateDescriptor:
        if not isinstance(parameter_layout, ParameterTreeLayout):
            raise TypeError("parameter_layout must be ParameterTreeLayout")
        return JaxOptimizerStateDescriptor(
            optimizer_id=self.optimizer_id,
            optimizer_capability="optimizer.jax_execution_v1",
            optimizer_schema_version="sgd_jax_state.v1",
            layout_digest=parameter_layout.digest(),
            state_keypaths=(
                ("step",),
                *(
                    ("per_parameter_steps", *entry.jax_keypath)
                    for entry in parameter_layout.entries
                ),
            ),
        )

    def initialize_jax_state(
        self,
        *,
        config: OptimizerConfig,
        parameter_layout: ParameterTreeLayout,
        optimizer_state: OptimizerState,
    ) -> Any:
        """Create the numerical SGD state under the existing optimizer identity."""

        self.validate_config(config)
        if optimizer_state.optimizer_id != self.optimizer_id:
            raise OptimizerContractError(
                "optimizer_jax_state_invalid",
                "optimizer state does not belong to SGD",
            )
        if optimizer_state.parameter_paths != parameter_layout.logical_paths:
            raise OptimizerContractError(
                "optimizer_jax_state_invalid",
                "optimizer state paths do not match the parameter layout",
            )
        try:
            jnp = import_module("jax.numpy")
        except ImportError as exc:
            raise OptimizerContractError(
                "optimizer_jax_capability_missing", "JAX is required for JAX SGD"
            ) from exc
        from radjax_student.optimizers.jax import JaxOptimizerState

        return JaxOptimizerState(
            envelope=optimizer_state,
            descriptor=self.jax_state_descriptor(parameter_layout),
            arrays={
                "step": jnp.asarray(optimizer_state.step, dtype=jnp.int32),
                "per_parameter_steps": parameter_layout.mapping_tree(
                    lambda _: jnp.asarray(0, dtype=jnp.int32)
                ),
            },
        )

    def apply_jax_updates(
        self,
        *,
        parameters: Any,
        gradients: Any,
        optimizer_array_state: Any,
        update_mask: Any,
        config: OptimizerConfig,
        schedule_values: dict[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Pure JAX SGD update; callers supply only a layout-derived mask."""

        self.validate_config(config)
        try:
            jax = import_module("jax")
            jnp = import_module("jax.numpy")
        except ImportError as exc:
            raise OptimizerContractError(
                "optimizer_jax_capability_missing", "JAX is required for JAX SGD"
            ) from exc
        learning_rate = schedule_values.get("learning_rate", config.learning_rate)
        learning_rate = jnp.asarray(learning_rate)
        learning_rate_valid = jnp.logical_and(
            jnp.all(jnp.isfinite(learning_rate)), jnp.all(learning_rate > 0)
        )
        gradient_leaves = jax.tree_util.tree_leaves(gradients)
        gradients_finite = jnp.asarray(True)
        for gradient in gradient_leaves:
            gradients_finite = jnp.logical_and(
                gradients_finite, jnp.all(jnp.isfinite(gradient))
            )
        updated_parameters = jax.tree_util.tree_map(
            lambda parameter, gradient, selected: jnp.where(
                selected, parameter - learning_rate * gradient, parameter
            ),
            parameters,
            gradients,
            update_mask,
        )
        changed_mask = jax.tree_util.tree_map(
            lambda parameter, updated, selected: jnp.logical_and(
                selected, jnp.any(parameter != updated)
            ),
            parameters,
            updated_parameters,
            update_mask,
        )
        per_parameter_steps = jax.tree_util.tree_map(
            lambda step, selected: jnp.where(selected, step + 1, step),
            optimizer_array_state["per_parameter_steps"],
            update_mask,
        )
        updated_array_state = {
            "step": optimizer_array_state["step"] + 1,
            "per_parameter_steps": per_parameter_steps,
        }
        return (
            updated_parameters,
            updated_array_state,
            changed_mask,
            {
                "learning_rate": learning_rate,
                "learning_rate_valid": learning_rate_valid,
                "gradients_finite": gradients_finite,
            },
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
