"""Static Mamba-2 parameter, state, objective, and HF schema."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from radjax_student.architecture.carry_descriptor import carry_descriptor_digest
from radjax_student.architecture.errors import ArchitectureIssue
from radjax_student.architecture.mamba2_reference.config import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
    MAMBA2_REFERENCE_CONTEXT_LENGTH,
    MAMBA2_REFERENCE_DTYPE,
    equation_shape,
    language_identities,
    reference_architecture_config,
    validate_mamba2_config,
)
from radjax_student.architecture.models import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureMetadata,
    IntermediateSurfaceDescriptor,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFParameterProjection,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)

STATIC_CAPABILITIES = (
    "architecture.batch_validation_v1",
    "architecture.objective.final_output_v1",
    "architecture.parameter_metadata_v1",
    "architecture.static_schema_v1",
    "architecture.update_scope.parameter_paths_v1",
    "architecture.update_scope.whole_student_v1",
)
INITIALIZATION_CAPABILITIES: tuple[str, ...] = STATIC_CAPABILITIES
EXECUTION_CAPABILITIES: tuple[str, ...] = STATIC_CAPABILITIES


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _parameter(
    path: str,
    shape: tuple[int, ...],
    role: str,
    regions: tuple[str, ...],
    *,
    tied_weight_group: str | None = None,
    source_path: str | None = None,
    mapping_class: str = "direct",
) -> ParameterDescriptor:
    return ParameterDescriptor(
        path=path,
        shape=shape,
        dtype=MAMBA2_REFERENCE_DTYPE,
        role=role,
        region_ids=("whole_student", *regions),
        metadata={
            "mapping_class": mapping_class,
            "source_path": source_path or path,
            **(
                {"tied_weight_group": tied_weight_group}
                if tied_weight_group is not None
                else {}
            ),
        },
    )


def parameter_catalog(config: ArchitectureConfig | None = None) -> ParameterCatalog:
    """Return a complete source-path mapping for the M2 reference profile."""

    config = reference_architecture_config() if config is None else config
    validate_mamba2_config(config)
    shape = equation_shape(config)
    parameters: list[ParameterDescriptor] = [
        _parameter(
            "backbone.embedding.weight",
            (shape.vocabulary_size, shape.d_model),
            "embedding",
            ("embedding",),
            tied_weight_group="token_embedding",
            source_path="backbone.embedding.weight",
        )
    ]
    for layer in range(shape.n_layer):
        prefix = f"backbone.layers.{layer}"
        region = (f"block_{layer}", f"mixer_{layer}")
        parameters.extend(
            (
                _parameter(
                    f"{prefix}.norm.weight",
                    (shape.d_model,),
                    "normalization",
                    region,
                    source_path="mamba_ssm.modules.block.Block.norm.weight",
                ),
                _parameter(
                    f"{prefix}.mixer.in_proj.weight",
                    (shape.d_in_proj, shape.d_model),
                    "recurrent_block",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.in_proj.weight",
                ),
                _parameter(
                    f"{prefix}.mixer.conv1d.weight",
                    (shape.conv_dim, 1, shape.d_conv),
                    "recurrent_block",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.conv1d.weight",
                ),
                _parameter(
                    f"{prefix}.mixer.conv1d.bias",
                    (shape.conv_dim,),
                    "recurrent_block",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.conv1d.bias",
                ),
                _parameter(
                    f"{prefix}.mixer.dt_bias",
                    (shape.nheads,),
                    "state_mixer",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.dt_bias",
                ),
                _parameter(
                    f"{prefix}.mixer.A_log",
                    (shape.nheads,),
                    "state_mixer",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.A_log",
                ),
                _parameter(
                    f"{prefix}.mixer.D",
                    (shape.nheads,),
                    "state_mixer",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.D",
                ),
                _parameter(
                    f"{prefix}.mixer.norm.weight",
                    (shape.d_ssm,),
                    "normalization",
                    region,
                    source_path="mamba_ssm.ops.triton.layernorm_gated.RMSNorm.weight",
                ),
                _parameter(
                    f"{prefix}.mixer.out_proj.weight",
                    (shape.d_model, shape.d_inner),
                    "recurrent_block",
                    region,
                    source_path="mamba_ssm.modules.mamba2.Mamba2.out_proj.weight",
                ),
            )
        )
    parameters.extend(
        (
            _parameter(
                "backbone.norm_f.weight",
                (shape.d_model,),
                "normalization",
                ("output",),
                source_path="mamba_ssm.models.mixer_seq_simple.MixerModel.norm_f.weight",
            ),
            _parameter(
                "lm_head.weight",
                (shape.vocabulary_size, shape.d_model),
                "output_head",
                ("head", "output"),
                tied_weight_group="token_embedding",
                source_path="mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel.lm_head.weight",
            ),
        )
    )
    return ParameterCatalog(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        parameters=tuple(parameters),
        metadata={
            "source_repository": "state-spaces/mamba",
            "source_pin": "95d8aba8a8c75aedcaa6143713b11e745e7cd0d9",
            "source_version": "v2.2.4",
            "weight_file_compatibility": False,
            "pretrained_weight_compatibility": False,
        },
    )


def initialization_parameter_slots(
    config: ArchitectureConfig | None = None,
) -> tuple[str, ...]:
    return parameter_catalog(config).paths


def parameter_layout(config: ArchitectureConfig | None = None) -> ParameterTreeLayout:
    catalog = parameter_catalog(config)
    return ParameterTreeLayout(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        entries=tuple(
            ParameterTreeLayoutEntry(
                logical_path=item.path,
                jax_keypath=tuple(item.path.split(".")),
                shape=item.shape,
                dtype=item.dtype,
                role=item.role,
                region_ids=item.region_ids,
                trainable=item.trainable_by_default,
                exportable=False,
                tied_weight_group=item.metadata.get("tied_weight_group"),
                metadata=dict(item.metadata),
            )
            for item in catalog.parameters
        ),
    )


def parameter_layout_for_vocabulary(
    vocabulary_size: int, *, context_length: int = MAMBA2_REFERENCE_CONTEXT_LENGTH
) -> ParameterTreeLayout:
    """Derive vocabulary-bearing layout without asserting language identity."""

    if (
        not isinstance(vocabulary_size, int)
        or isinstance(vocabulary_size, bool)
        or vocabulary_size <= 0
    ):
        raise ValueError("vocabulary_size must be positive")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise ValueError("context_length must be positive")
    descriptors = list(parameter_catalog().parameters)
    for index, descriptor in enumerate(descriptors):
        if descriptor.path in {
            "backbone.embedding.weight",
            "lm_head.weight",
        }:
            descriptors[index] = ParameterDescriptor(
                path=descriptor.path,
                shape=(vocabulary_size, descriptor.shape[1]),
                dtype=descriptor.dtype,
                role=descriptor.role,
                region_ids=descriptor.region_ids,
                metadata=dict(descriptor.metadata),
            )
    catalog = ParameterCatalog(MAMBA2_REFERENCE_ARCHITECTURE_ID, tuple(descriptors))
    return ParameterTreeLayout(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        entries=tuple(
            ParameterTreeLayoutEntry(
                logical_path=item.path,
                jax_keypath=tuple(item.path.split(".")),
                shape=item.shape,
                dtype=item.dtype,
                role=item.role,
                region_ids=item.region_ids,
                trainable=item.trainable_by_default,
                exportable=False,
                tied_weight_group=item.metadata.get("tied_weight_group"),
                metadata=dict(item.metadata),
            )
            for item in catalog.parameters
        ),
    )


def carry_descriptor(config: ArchitectureConfig | None = None) -> dict[str, Any]:
    config = reference_architecture_config() if config is None else config
    validate_mamba2_config(config)
    shape = equation_shape(config)
    persistent: dict[str, dict[str, Any]] = {}
    for layer in range(shape.n_layer):
        persistent[f"layers.{layer}.conv_state"] = {
            "dtype": MAMBA2_REFERENCE_DTYPE,
            "shape": [shape.batch_size, shape.conv_dim, shape.d_conv],
        }
        persistent[f"layers.{layer}.ssm_state"] = {
            "dtype": MAMBA2_REFERENCE_DTYPE,
            "shape": [shape.batch_size, shape.nheads, shape.headdim, shape.d_state],
        }
    return {
        "schema_version": "radjax.mamba2_reference_carry.v1",
        "persistent_leaves": persistent,
        "nonpersistent_token_local_values": ["z", "x", "B", "C", "dt", "y"],
        "state_semantics": {
            "conv_state": "depthwise convolution history across tokens/chunks",
            "ssm_state": "selective state-space recurrence across tokens/chunks",
            "token_local": "projection and output intermediates do not persist",
            "student_step_boundary": (
                "carry crosses learning steps; gradients stop at boundary"
            ),
        },
    }


def carry_descriptor_digest_for_config(config: ArchitectureConfig | None = None) -> str:
    return carry_descriptor_digest(carry_descriptor(config))


def capability_profile() -> ArchitectureCapabilityProfile:
    return ArchitectureCapabilityProfile(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        version=MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
        capabilities=STATIC_CAPABILITIES,
        non_capabilities=(
            "architecture.parameter_initialization_v1",
            "architecture.jax_execution_v1",
        ),
        metadata={"phase": "M2.2", "static_schema_only": True},
    )


def architecture_metadata(
    config: ArchitectureConfig | None = None,
) -> ArchitectureMetadata:
    config = reference_architecture_config() if config is None else config
    validate_mamba2_config(config)
    catalog = parameter_catalog(config)
    regions = [NamedRegion("whole_student", catalog.trainable_paths)]
    regions.extend(
        NamedRegion(
            f"block_{layer}",
            tuple(
                path
                for path in catalog.paths
                if path.startswith(f"backbone.layers.{layer}.")
            ),
        )
        for layer in range(equation_shape(config).n_layer)
    )
    regions.extend(
        (
            NamedRegion("embedding", ("backbone.embedding.weight",)),
            NamedRegion("head", ("lm_head.weight",)),
            NamedRegion(
                "output",
                ("backbone.norm_f.weight", "lm_head.weight"),
            ),
        )
    )
    return ArchitectureMetadata(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        parameter_catalog=catalog,
        capability_profile=capability_profile(),
        named_regions=tuple(regions),
        objective_surfaces=(
            IntermediateSurfaceDescriptor(
                surface_id="final_output",
                kind="logits",
                shape_contract={"rank": 3, "shape": ["B", "T", "V"]},
                available_in_training=False,
                available_in_inference=False,
            ),
        ),
        warnings=(
            ArchitectureIssue(
                code="mamba2_reference_static_schema_only",
                message=(
                    "M2.2 declares schema and identity only; no initialization or "
                    "execution is available."
                ),
            ),
        ),
        claims_not_made=(
            "equation_parity_not_yet_executed",
            "initialization_not_implemented",
            "forward_execution_not_proven",
            "weight_file_compatibility_not_claimed",
            "transformers_compatibility_not_claimed",
        ),
    )


def hf_descriptor(config: ArchitectureConfig) -> HFCompatibilityDescriptor:
    validate_mamba2_config(config)
    shape = equation_shape(config)
    catalog = parameter_catalog(config)
    layout = parameter_layout(config)
    identities = language_identities(config)
    if identities is None:
        tokenizer = HFTokenizerIdentity(
            "mamba2_reference_fixture_tokenizer",
            "not_claimed",
            _digest({"tokenizer": "not_claimed"}),
            _digest({"config": "not_claimed"}),
            "fixture_only",
            _digest({"normalization": "not_claimed"}),
            "synthetic",
        )
        vocabulary = HFVocabularyIdentity(
            shape.vocabulary_size,
            _digest({"fixture_vocabulary_size": shape.vocabulary_size}),
            _digest({"fixture_token_mapping": "not_claimed"}),
            _digest({"added_tokens": []}),
            None,
        )
        special_tokens = HFSpecialTokenIdentity(None, None, None, None, None)
    else:
        tokenizer, vocabulary, special_tokens = identities
    return HFCompatibilityDescriptor(
        schema_version="hf_compatibility_descriptor.v2",
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        architecture_plugin_version=MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
        model_type="mamba2_reference",
        architecture_config_digest=_digest(config.to_dict()),
        parameter_catalog_digest=_digest(catalog.to_dict()),
        parameter_layout_digest=layout.digest(),
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        special_tokens=special_tokens,
        parameter_projections=tuple(
            HFParameterProjection(
                entry.logical_path,
                entry.jax_keypath,
                entry.shape,
                entry.dtype,
                "non_exportable",
                None,
                "identity",
                entry.tied_weight_group,
                "weight_file_compatibility_not_claimed",
            )
            for entry in layout.entries
        ),
        architecture_projection=HFArchitectureProjection(
            "mamba2_reference_config",
            "mamba2_reference",
            shape.d_model,
            shape.n_layer,
            vocabulary.vocabulary_size,
            config.sequence_length,
            dict(config.model_config),
        ),
        non_claims=(
            "from_pretrained_not_implemented",
            "hf_conversion_not_implemented",
            "save_pretrained_not_implemented",
            "weight_file_compatibility_not_claimed",
            "transformers_compatibility_not_claimed",
        ),
        notes=(
            "M2.2 architecture-owned descriptor; no HF conversion or weight-file "
            "support."
        ),
    )


__all__ = [
    "STATIC_CAPABILITIES",
    "INITIALIZATION_CAPABILITIES",
    "EXECUTION_CAPABILITIES",
    "architecture_metadata",
    "capability_profile",
    "carry_descriptor",
    "carry_descriptor_digest_for_config",
    "hf_descriptor",
    "initialization_parameter_slots",
    "parameter_catalog",
    "parameter_layout",
    "parameter_layout_for_vocabulary",
]
