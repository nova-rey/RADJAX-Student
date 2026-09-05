"""Mamba-2 reference plugin boundary with lazy pure-JAX execution.

The package intentionally advertises inspection and validation only until the
initialization and pure-JAX execution checkpoints are accepted.  Keeping the
identity registered at this stage lets callers fail closed on unsupported
capabilities instead of mistaking a partial plugin for an executable model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.architecture.carry_descriptor import (
    carry_descriptor_digest,
    describe_mapping_carry,
)
from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
from radjax_student.architecture.mamba2_reference.config import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
    reference_architecture_config,
    validate_mamba2_config,
)
from radjax_student.architecture.mamba2_reference.schema import (
    architecture_metadata,
    capability_profile,
    carry_descriptor,
    hf_descriptor,
    parameter_catalog,
    parameter_layout,
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
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


@dataclass(frozen=True)
class Mamba2ReferencePlugin:
    """Explicit Mamba-2 identity with static-only M2.2 capability."""

    architecture_id: str = MAMBA2_REFERENCE_ARCHITECTURE_ID
    architecture_version: int = MAMBA2_REFERENCE_ARCHITECTURE_VERSION

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return capability_profile("execution")

    def validate_config(self, config: ArchitectureConfig) -> None:
        validate_mamba2_config(config)

    def describe_parameters(
        self,
        parameters: object | None = None,
        *,
        architecture_config: ArchitectureConfig | None = None,
    ) -> ParameterCatalog:
        config = (
            reference_architecture_config()
            if architecture_config is None
            else architecture_config
        )
        self.validate_config(config)
        catalog = parameter_catalog(config)
        if parameters is not None:
            try:
                parameter_layout(config).validate_materialized_parameters(parameters)
            except (TypeError, ValueError) as exc:
                raise ArchitectureContractError(
                    "architecture_parameter_catalog_invalid",
                    "materialized parameters do not match the Mamba-2 static layout",
                ) from exc
        return catalog

    def architecture_metadata(self) -> ArchitectureMetadata:
        return architecture_metadata(stage="execution")

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        self.validate_config(request.config)
        if request.precision_policy != "float32":
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "Mamba-2 reference initialization requires float32 precision",
            )
        initialization_key = request.runtime_initialization_material
        if initialization_key is None:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "runtime-supplied initialization material is required",
            )
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "JAX initialization support is unavailable",
            ) from exc
        catalog = parameter_catalog(request.config)
        layout = parameter_layout(request.config)
        try:
            keys = jax.random.split(initialization_key, len(catalog.paths))
        except (TypeError, ValueError) as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "runtime-supplied initialization material is invalid",
            ) from exc
        keys_by_path = dict(zip(catalog.paths, keys, strict=True))
        parameters = layout.mapping_tree(
            lambda entry: self._initialize_leaf(
                jax, jnp, entry, keys_by_path[entry.logical_path]
            )
        )
        # The source model ties lm_head.weight to the token embedding.  Preserve
        # that relation in the materialized tree, rather than merely recording a
        # descriptive tied group in the static layout.
        parameters["lm_head"]["weight"] = parameters["backbone"]["embedding"]["weight"]
        try:
            layout.validate_materialized_parameters(parameters)
        except (TypeError, ValueError) as exc:
            raise ArchitectureContractError(
                "architecture_initialization_failed",
                "Mamba-2 initialization did not satisfy its parameter layout",
            ) from exc
        carry = self._zeroed_carry(jnp, request.config)
        descriptor = carry_descriptor(request.config)
        state_id = "mamba2_reference_state.v1"
        return ArchitectureInitResult(
            parameter_catalog=catalog,
            architecture_state=ArchitectureState(
                state_id,
                metadata={"carry_schema_version": descriptor["schema_version"]},
            ),
            parameters=parameters,
            architecture_carry=carry,
            architecture_carry_descriptor={
                "schema_version": "architecture_carry.v1",
                "state_id": state_id,
                "pytree_descriptor_digest": carry_descriptor_digest(
                    describe_mapping_carry(carry)
                ),
            },
            parameter_layout=layout,
            hf_descriptor=hf_descriptor(request.config),
            claims_not_made=(
                "initialization_parity_not_claimed",
                "upstream_weight_loading_not_claimed",
                "pretrained_model_equivalence_not_claimed",
                "training_recipe_parity_not_claimed",
                "optimized_kernel_compatibility_not_claimed",
            ),
        )

    @staticmethod
    def _initialize_leaf(
        jax: object, jnp: object, entry: object, key: object
    ) -> object:
        """Deterministic float32 values; source-specific init parity is unclaimed."""

        path = entry.logical_path
        if path.endswith(".conv1d.bias"):
            return jnp.zeros(entry.shape, dtype=jnp.float32)
        if path.endswith(".mixer.D"):
            return jnp.ones(entry.shape, dtype=jnp.float32)
        if path.endswith(".mixer.A_log"):
            return jnp.log(jnp.arange(1, entry.shape[0] + 1, dtype=jnp.float32))
        if path.endswith(".mixer.dt_bias"):
            return jnp.zeros(entry.shape, dtype=jnp.float32)
        return jax.random.normal(key, entry.shape, dtype=jnp.float32) * jnp.asarray(
            0.02, dtype=jnp.float32
        )

    @staticmethod
    def _zeroed_carry(jnp: object, config: ArchitectureConfig) -> dict[str, object]:
        persistent = carry_descriptor(config)["persistent_leaves"]
        if not isinstance(persistent, Mapping):
            raise ArchitectureContractError(
                "architecture_internal_error", "Mamba-2 carry descriptor is invalid"
            )
        carry = {
            name: jnp.zeros(tuple(spec["shape"]), dtype=jnp.float32)
            for name, spec in persistent.items()
        }
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
        valid = (
            isinstance(token_ids, (list, tuple))
            and len(token_ids) == 1
            and isinstance(token_ids[0], (list, tuple))
            and 1 <= len(token_ids[0]) <= config.sequence_length
            and all(
                isinstance(token, int)
                and not isinstance(token, bool)
                and 0 <= token < config.vocab_size
                for token in token_ids[0]
            )
        )
        if not valid:
            return BatchValidationResult(
                status="fail",
                blockers=(
                    ArchitectureIssue(
                        code="architecture_batch_incompatible",
                        message=(
                            "Mamba-2 requires one rank-2 integer token sequence "
                            "whose length does not exceed the configured context"
                        ),
                    ),
                ),
            )
        return BatchValidationResult(status="pass")

    def forward(self, request: ForwardRequest) -> ForwardResult:
        del request
        raise ArchitectureContractError(
            "architecture_forward_failed",
            "Mamba-2 JAX execution is unavailable before M2.4",
        )

    def apply_jax(
        self,
        parameters: object,
        architecture_state: object,
        batch: object,
        *,
        architecture_config: ArchitectureConfig | None = None,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: object | None,
    ) -> ForwardResult:
        del training, rng_key
        if objective_scope.kind != "final_output":
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "Mamba-2 JAX execution exposes only the final logits surface",
            )
        config = (
            reference_architecture_config()
            if architecture_config is None
            else architecture_config
        )
        self.validate_config(config)
        try:
            layout = parameter_layout(config)
            layout.validate_materialized_parameters(parameters)
            self._validate_carry(architecture_state, config)
            token_ids = self._validate_jax_tokens(
                batch,
                vocabulary_size=config.vocab_size,
                context_length=config.sequence_length,
            )
            from radjax_student.architecture.mamba2_reference.kernels import (
                mamba2_sequence,
            )

            logits, carry = mamba2_sequence(
                parameters, token_ids[0], architecture_state
            )
        except ArchitectureContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "Mamba-2 JAX execution received invalid values",
            ) from exc
        return ForwardResult(
            outputs=logits[None, :, :],
            updated_architecture_carry=carry,
            claims_not_made=(
                "optimized_kernel_compatibility_not_claimed",
                "pretrained_model_equivalence_not_claimed",
                "training_recipe_parity_not_claimed",
                "weight_file_compatibility_not_claimed",
                "cross_step_bptt_not_claimed",
            ),
        )

    @staticmethod
    def _validate_carry(carry: object, config: ArchitectureConfig) -> None:
        descriptor = carry_descriptor(config)["persistent_leaves"]
        if not isinstance(carry, Mapping) or set(carry) != set(descriptor):
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "Mamba-2 carry does not match its persistent descriptor",
            )
        for name, specification in descriptor.items():
            value = carry[name]
            if (
                tuple(getattr(value, "shape", ())) != tuple(specification["shape"])
                or str(getattr(value, "dtype", "")) != specification["dtype"]
            ):
                raise ArchitectureContractError(
                    "architecture_forward_failed",
                    "Mamba-2 carry does not match its persistent descriptor",
                )

    @staticmethod
    def _validate_jax_tokens(
        batch: object,
        *,
        vocabulary_size: int,
        context_length: int,
    ) -> object:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ArchitectureContractError(
                "architecture_forward_failed",
                "JAX execution support is unavailable",
            ) from exc
        inputs = getattr(batch, "inputs", None)
        token_ids = inputs.get("token_ids") if isinstance(inputs, Mapping) else None
        if (
            getattr(token_ids, "ndim", None) != 2
            or token_ids.shape[0] != 1
            or not 1 <= token_ids.shape[1] <= context_length
            or not jnp.issubdtype(token_ids.dtype, jnp.integer)
        ):
            raise ArchitectureContractError(
                "architecture_batch_incompatible",
                "Mamba-2 requires one rank-2 integer token sequence",
            )
        if not isinstance(token_ids, jax.core.Tracer) and bool(
            jnp.any((token_ids < 0) | (token_ids >= vocabulary_size))
        ):
            raise ArchitectureContractError(
                "architecture_batch_incompatible",
                "Mamba-2 token_ids are outside the configured vocabulary",
            )
        return token_ids

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection:
        if parameter_catalog.architecture_id != self.architecture_id:
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter catalog does not match the Mamba-2 static schema",
            )
        if scope.kind == "whole_student":
            selected = parameter_catalog.trainable_paths
        elif scope.kind == "parameter_paths":
            selected = scope.parameter_paths
            unknown = sorted(set(selected) - set(parameter_catalog.paths))
            if unknown:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown Mamba-2 parameter path",
                    details={"unknown_paths": unknown},
                )
        elif scope.kind == "named_region":
            region_id = scope.region_id
            region_paths = {
                path
                for path in parameter_catalog.paths
                if region_id in parameter_catalog.get(path).region_ids
            }
            if not region_paths:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown Mamba-2 named region",
                    details={"region_id": region_id},
                )
            selected = tuple(
                path for path in parameter_catalog.paths if path in region_paths
            )
        else:
            raise ArchitectureContractError(
                "architecture_update_scope_unsupported",
                "Mamba-2 supports whole-student, named-region, and explicit-path "
                "scopes",
            )
        return ResolvedUpdateSelection(
            selection_id=f"{self.architecture_id}:{scope.kind}",
            selected_parameter_paths=tuple(selected),
            excluded_parameter_paths=tuple(
                path for path in parameter_catalog.paths if path not in selected
            ),
            capabilities=(f"architecture.update_scope.{scope.kind}_v1",),
            metadata={"phase": "M2.4", "jax_execution_available": True},
        )

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection:
        if (
            metadata.architecture_id != self.architecture_id
            or scope.kind != "final_output"
        ):
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "Mamba-2 exposes only the final logits objective surface",
            )
        return ResolvedObjectiveSelection(
            scope=scope,
            surface_id="final_output",
            surface_role="logits",
            required_capabilities=("architecture.objective.final_output_v1",),
            metadata={"phase": "M2.4", "jax_execution_available": True},
        )

    def hf_compatibility_descriptor(
        self, request: ArchitectureInitRequest, result: ArchitectureInitResult
    ) -> HFCompatibilityDescriptor:
        self.validate_config(request.config)
        expected_catalog = parameter_catalog(request.config)
        expected_layout = parameter_layout(request.config)
        if (
            result.parameter_catalog != expected_catalog
            or result.parameter_layout != expected_layout
        ):
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "HF projection must use the declared Mamba-2 static schema",
            )
        return hf_descriptor(request.config)


__all__ = ["Mamba2ReferencePlugin"]
