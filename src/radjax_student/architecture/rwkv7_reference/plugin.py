"""Tiny-domain pinned-equation RWKV-7 reference plugin for P4.4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
from radjax_student.architecture.models import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitectureMetadata,
    ArchitectureState,
    BatchValidationResult,
    ForwardRequest,
    ForwardResult,
    ParameterCatalog,
    ResolvedObjectiveSelection,
)
from radjax_student.architecture.rwkv7_reference.config import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7_REFERENCE_CONTEXT_LENGTH,
    RWKV7_REFERENCE_VOCABULARY_SIZE,
    validate_reference_config,
)
from radjax_student.architecture.rwkv7_reference.schema import (
    architecture_metadata,
    capability_profile,
    carry_descriptor,
    hf_descriptor,
    initialization_parameter_slots,
    parameter_catalog,
    parameter_layout,
)
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


@dataclass(frozen=True)
class RWKV7ReferencePlugin:
    """Frozen tiny-domain RWKV-7 plugin with lazy JAX execution."""

    architecture_id: str = RWKV7_REFERENCE_ARCHITECTURE_ID
    architecture_version: int = RWKV7_REFERENCE_ARCHITECTURE_VERSION

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return capability_profile()

    def validate_config(self, config: ArchitectureConfig) -> None:
        validate_reference_config(config)

    def describe_parameters(self, parameters: object | None = None) -> ParameterCatalog:
        if parameters is not None:
            try:
                parameter_layout().validate_materialized_parameters(parameters)
            except (TypeError, ValueError) as exc:
                raise ArchitectureContractError(
                    "architecture_parameter_catalog_invalid",
                    "materialized parameters do not match the RWKV-7 layout",
                ) from exc
        return parameter_catalog()

    def architecture_metadata(self) -> ArchitectureMetadata:
        return architecture_metadata()

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        self.validate_config(request.config)
        if request.precision_policy != "float32":
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "RWKV-7 reference initialization requires float32 precision",
            )
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "JAX initialization support is unavailable",
            ) from exc

        initialization_key = request.runtime_initialization_material
        if initialization_key is None:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "runtime-supplied initialization material is required",
            )

        catalog = parameter_catalog()
        layout = parameter_layout()
        slots = initialization_parameter_slots()
        if slots != catalog.paths:
            raise ArchitectureContractError(
                "architecture_internal_error",
                "RWKV-7 initialization slots do not match the parameter catalog",
            )
        try:
            keys = jax.random.split(initialization_key, len(slots))
        except (TypeError, ValueError) as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "runtime-supplied initialization material is invalid",
            ) from exc
        keys_by_path = dict(zip(slots, keys, strict=True))
        scale = jnp.asarray(0.02, dtype=jnp.float32)
        parameters = layout.mapping_tree(
            lambda entry: (
                jax.random.normal(
                    keys_by_path[entry.logical_path], entry.shape, dtype=jnp.float32
                )
                * scale
            )
        )
        try:
            layout.validate_materialized_parameters(parameters)
        except ValueError as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "RWKV-7 initialization did not satisfy its parameter layout",
            ) from exc
        carry = self._zeroed_carry(jnp)
        descriptor = carry_descriptor()
        return ArchitectureInitResult(
            parameter_catalog=catalog,
            architecture_state=ArchitectureState(
                "rwkv7_reference_state.v1",
                metadata={"carry_schema_version": descriptor["schema_version"]},
            ),
            parameters=parameters,
            architecture_carry=carry,
            architecture_carry_descriptor=descriptor,
            parameter_layout=layout,
            hf_descriptor=hf_descriptor(request.config),
            claims_not_made=(
                "equation_parity_outside_fixture_domain_not_claimed",
                "gradient_not_computed",
                "initialization_parity_not_claimed",
                "optimizer_not_invoked",
                "training_loop_not_run",
                "weight_file_compatibility_not_claimed",
            ),
        )

    @staticmethod
    def _zeroed_carry(jnp: object) -> dict[str, object]:
        descriptor = carry_descriptor()["persistent_leaves"]
        if not isinstance(descriptor, Mapping):
            raise ArchitectureContractError(
                "architecture_internal_error", "RWKV-7 carry descriptor is invalid"
            )
        carry = {
            name: jnp.zeros(tuple(specification["shape"]), dtype=jnp.float32)
            for name, specification in descriptor.items()
        }
        for name, specification in descriptor.items():
            value = carry[name]
            if (
                tuple(value.shape) != tuple(specification["shape"])
                or str(value.dtype) != specification["dtype"]
            ):
                raise ArchitectureContractError(
                    "architecture_initialization_failed",
                    "RWKV-7 carry does not match its declared descriptor",
                )
        return carry

    def validate_batch(
        self, batch: LearningBatch, config: ArchitectureConfig
    ) -> BatchValidationResult:
        self.validate_config(config)
        if not isinstance(batch, LearningBatch):
            raise ArchitectureContractError(
                "architecture_batch_incompatible", "batch must be LearningBatch"
            )
        token_ids = batch.inputs.get("token_ids")
        if not isinstance(token_ids, Mapping) or token_ids.get("rank") != 2:
            return BatchValidationResult(
                status="fail",
                blockers=(
                    ArchitectureIssue(
                        code="architecture_batch_incompatible",
                        message="RWKV-7 reference expects rank-2 token_ids metadata",
                    ),
                ),
            )
        return BatchValidationResult(status="pass")

    def forward(self, request: ForwardRequest) -> ForwardResult:
        del request
        raise ArchitectureContractError(
            "architecture_forward_failed",
            "RWKV-7 JAX execution requires the apply_jax architecture boundary",
        )

    def apply_jax(
        self,
        parameters: object,
        architecture_state: object,
        batch: object,
        *,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: object | None,
    ) -> ForwardResult:
        """Execute the pinned tiny-domain inference equations through JAX only."""

        del training, rng_key
        if not isinstance(objective_scope, ObjectiveScope) or objective_scope.kind != (
            "final_output"
        ):
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "RWKV-7 JAX execution exposes only the final logits surface",
            )
        try:
            parameter_layout().validate_materialized_parameters(parameters)
            self._validate_carry(architecture_state)
            token_ids = self._validate_jax_tokens(batch)
            from radjax_student.architecture.rwkv7_reference.kernels import (
                rwkv7_sequence,
            )

            logits, carry = rwkv7_sequence(parameters, token_ids[0], architecture_state)
        except ArchitectureContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "RWKV-7 JAX execution received invalid values",
            ) from exc
        return ForwardResult(
            outputs=logits[None, :, :],
            updated_architecture_carry=carry,
            claims_not_made=(
                "equation_parity_outside_fixture_domain_not_claimed",
                "initialization_parity_not_claimed",
                "training_recipe_parity_not_claimed",
                "weight_file_compatibility_not_claimed",
            ),
        )

    @staticmethod
    def _validate_carry(carry: object) -> None:
        descriptor = carry_descriptor()["persistent_leaves"]
        if not isinstance(carry, Mapping) or set(carry) != set(descriptor):
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "RWKV-7 carry does not match its persistent descriptor",
            )
        for name, specification in descriptor.items():
            value = carry[name]
            if (
                tuple(getattr(value, "shape", ())) != tuple(specification["shape"])
                or str(getattr(value, "dtype", "")) != specification["dtype"]
            ):
                raise ArchitectureContractError(
                    "architecture_forward_failed",
                    "RWKV-7 carry does not match its persistent descriptor",
                )

    @staticmethod
    def _validate_jax_tokens(batch: object) -> object:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "JAX execution support is unavailable",
            ) from exc
        inputs = getattr(batch, "inputs", None)
        if not isinstance(inputs, Mapping) or "token_ids" not in inputs:
            raise ArchitectureContractError(
                "architecture_batch_incompatible",
                "RWKV-7 JAX execution requires token_ids",
            )
        token_ids = inputs["token_ids"]
        if (
            getattr(token_ids, "ndim", None) != 2
            or token_ids.shape[0] != 1
            or token_ids.shape[1] != RWKV7_REFERENCE_CONTEXT_LENGTH
            or not jnp.issubdtype(token_ids.dtype, jnp.integer)
        ):
            raise ArchitectureContractError(
                "architecture_batch_incompatible",
                "RWKV-7 requires one rank-2 integer tiny-domain token sequence",
            )
        if not isinstance(token_ids, jax.core.Tracer) and bool(
            jnp.any((token_ids < 0) | (token_ids >= RWKV7_REFERENCE_VOCABULARY_SIZE))
        ):
            raise ArchitectureContractError(
                "architecture_batch_incompatible",
                "RWKV-7 token_ids must be within the frozen vocabulary",
            )
        return token_ids

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection:
        if parameter_catalog != self.describe_parameters():
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter catalog does not match the RWKV-7 static schema",
            )
        if scope.kind == "whole_student":
            selected = parameter_catalog.trainable_paths
        elif scope.kind == "parameter_paths":
            selected = scope.parameter_paths
            unknown = sorted(set(selected) - set(parameter_catalog.paths))
            if unknown:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown RWKV-7 parameter path",
                    details={"unknown_paths": unknown},
                )
        else:
            raise ArchitectureContractError(
                "architecture_update_scope_unsupported",
                "P4.4 supports only whole-student or explicit-path update scopes",
            )
        return ResolvedUpdateSelection(
            selection_id=f"{self.architecture_id}:{scope.kind}",
            selected_parameter_paths=tuple(selected),
            excluded_parameter_paths=tuple(
                path for path in parameter_catalog.paths if path not in selected
            ),
            capabilities=(f"architecture.update_scope.{scope.kind}_v1",),
            metadata={"phase": "P4.4", "jax_execution_available": True},
        )

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection:
        if metadata != self.architecture_metadata() or scope.kind != "final_output":
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "P4.4 declares only the final logits objective surface",
            )
        return ResolvedObjectiveSelection(
            scope=scope,
            surface_id="final_output",
            surface_role="logits",
            required_capabilities=("architecture.objective.final_output_v1",),
            metadata={"phase": "P4.4", "jax_execution_available": True},
        )

    def hf_compatibility_descriptor(
        self, request: ArchitectureInitRequest, result: ArchitectureInitResult
    ) -> HFCompatibilityDescriptor:
        self.validate_config(request.config)
        if (
            result.parameter_catalog != parameter_catalog()
            or result.parameter_layout != parameter_layout()
        ):
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "HF projection must use the declared RWKV-7 static schema",
            )
        return hf_descriptor(request.config)


__all__ = ["RWKV7ReferencePlugin"]
