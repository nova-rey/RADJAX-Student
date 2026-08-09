"""Static parameter, carry, objective, and HF schema for RWKV-7 reference."""

from __future__ import annotations

import hashlib
import json

from radjax_student.architecture.errors import ArchitectureIssue
from radjax_student.architecture.models import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureMetadata,
    IntermediateSurfaceDescriptor,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.architecture.rwkv7_reference.config import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7_REFERENCE_DTYPE,
    RWKV7_REFERENCE_HIDDEN_SIZE,
    RWKV7_REFERENCE_LAYER_COUNT,
    RWKV7_REFERENCE_VOCABULARY_SIZE,
    equation_shape,
    language_identities,
    reference_architecture_config,
    validate_rwkv7_config,
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

INITIALIZATION_CAPABILITIES = (
    *STATIC_CAPABILITIES,
    "architecture.parameter_initialization_v1",
)

EXECUTION_CAPABILITIES = (
    *INITIALIZATION_CAPABILITIES,
    "architecture.jax_execution_v1",
)
CARRY_PYTREE_DESCRIPTOR_DIGEST = (
    "f44941bc4bb4becd2cd234c390889d138f7f7feceeaf754eb232368e8625375a"
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _parameter(
    path: str, shape: tuple[int, ...], role: str, regions: tuple[str, ...]
) -> ParameterDescriptor:
    return ParameterDescriptor(
        path=path,
        shape=shape,
        dtype=RWKV7_REFERENCE_DTYPE,
        role=role,
        region_ids=("whole_student", *regions),
        metadata={
            "representation": _representation(path),
        },
    )


def _representation(path: str) -> str:
    if path.endswith(".att.r_k"):
        return "pinned_numpy_flat_to_head_matrix"
    if path.endswith((".att.x_r", ".att.x_w", ".att.x_k", ".att.x_v")):
        return "pinned_numpy_squeeze_to_vector"
    if path.endswith((".att.x_a", ".att.x_g", ".att.w0", ".att.a0")):
        return "pinned_numpy_squeeze_to_vector"
    if path.endswith((".att.v0", ".att.k_k", ".att.k_a", ".ffn.x_k")):
        return "pinned_numpy_squeeze_to_vector"
    return "direct_pinned_numpy_array"


def parameter_catalog(config: ArchitectureConfig | None = None) -> ParameterCatalog:
    """Describe every parameter used by the pinned tiny-domain equations."""

    if config is None:
        config = reference_architecture_config()
    validate_rwkv7_config(config)
    shape = equation_shape(config)
    vocabulary_size = shape.vocabulary_size
    hidden = shape.hidden_size
    parameters = [
        _parameter(
            "emb.weight",
            (vocabulary_size, hidden),
            "embedding",
            ("embedding",),
        ),
        _parameter("blocks.0.ln0.weight", (hidden,), "normalization", ("block_0",)),
        _parameter("blocks.0.ln0.bias", (hidden,), "normalization", ("block_0",)),
    ]
    for block in range(shape.layer_count):
        region = (f"block_{block}",)
        prefix = f"blocks.{block}"
        for norm in ("ln1", "ln2"):
            parameters.extend(
                (
                    _parameter(
                        f"{prefix}.{norm}.weight", (hidden,), "normalization", region
                    ),
                    _parameter(
                        f"{prefix}.{norm}.bias", (hidden,), "normalization", region
                    ),
                )
            )
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            parameters.append(
                _parameter(f"{prefix}.att.{name}", (hidden,), "attention_block", region)
            )
        parameters.extend(
            (
                _parameter(f"{prefix}.att.w0", (hidden,), "attention_block", region),
                _parameter(
                    f"{prefix}.att.w1",
                    (hidden, shape.time_decay_rank),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.w2",
                    (shape.time_decay_rank, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(f"{prefix}.att.a0", (hidden,), "attention_block", region),
                _parameter(
                    f"{prefix}.att.a1",
                    (hidden, shape.time_aaa_rank),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.a2",
                    (shape.time_aaa_rank, hidden),
                    "attention_block",
                    region,
                ),
            )
        )
        if block > 0:
            parameters.extend(
                (
                    _parameter(
                        f"{prefix}.att.v0", (hidden,), "attention_block", region
                    ),
                    _parameter(
                        f"{prefix}.att.v1",
                        (hidden, shape.time_value_rank),
                        "attention_block",
                        region,
                    ),
                    _parameter(
                        f"{prefix}.att.v2",
                        (shape.time_value_rank, hidden),
                        "attention_block",
                        region,
                    ),
                )
            )
        parameters.extend(
            (
                _parameter(
                    f"{prefix}.att.g1",
                    (hidden, shape.time_gate_rank),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.g2",
                    (shape.time_gate_rank, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(f"{prefix}.att.k_k", (hidden,), "attention_block", region),
                _parameter(f"{prefix}.att.k_a", (hidden,), "attention_block", region),
                _parameter(
                    f"{prefix}.att.r_k",
                    (shape.head_count, shape.head_size),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.receptance.weight",
                    (hidden, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.key.weight",
                    (hidden, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.value.weight",
                    (hidden, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.output.weight",
                    (hidden, hidden),
                    "attention_block",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.ln_x.weight",
                    (hidden,),
                    "normalization",
                    region,
                ),
                _parameter(
                    f"{prefix}.att.ln_x.bias", (hidden,), "normalization", region
                ),
                _parameter(f"{prefix}.ffn.x_k", (hidden,), "channel_mixer", region),
                _parameter(
                    f"{prefix}.ffn.key.weight",
                    (shape.ffn_width, hidden),
                    "channel_mixer",
                    region,
                ),
                _parameter(
                    f"{prefix}.ffn.value.weight",
                    (hidden, shape.ffn_width),
                    "channel_mixer",
                    region,
                ),
            )
        )
    parameters.extend(
        (
            _parameter("ln_out.weight", (hidden,), "normalization", ("output",)),
            _parameter("ln_out.bias", (hidden,), "normalization", ("output",)),
            _parameter(
                "head.weight",
                (vocabulary_size, hidden),
                "output_head",
                ("head", "output"),
            ),
        )
    )
    return ParameterCatalog(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        parameters=tuple(parameters),
        metadata={
            "pinned_source": "BlinkDL/RWKV-LM:RWKV-v7/rwkv_v7_numpy.py",
            "weight_file_compatibility": False,
        },
    )


def pinned_numpy_parameter_order(
    config: ArchitectureConfig | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the reviewed source-prefix order consumed by the pinned NumPy code."""

    time_mix_prefix = (
        "x_r",
        "x_w",
        "x_k",
        "x_v",
        "x_a",
        "x_g",
        "w0",
        "r_k",
        "w1",
        "w2",
        "a1",
        "a2",
        "a0",
        "g1",
        "g2",
    )
    time_mix_suffix = (
        "k_k",
        "k_a",
        "receptance.weight",
        "key.weight",
        "value.weight",
        "output.weight",
        "ln_x.weight",
        "ln_x.bias",
    )
    order = {
        "emb": ("emb.weight",),
        "blocks.0.ln0": ("blocks.0.ln0.weight", "blocks.0.ln0.bias"),
    }
    shape = equation_shape(config or reference_architecture_config())
    for block in range(shape.layer_count):
        prefix = f"blocks.{block}"
        order[f"{prefix}.ln1"] = (
            f"{prefix}.ln1.weight",
            f"{prefix}.ln1.bias",
        )
        order[f"{prefix}.att"] = tuple(
            f"{prefix}.att.{name}"
            for name in (
                *time_mix_prefix,
                *(("v2", "v1", "v0") if block > 0 else ()),
                *time_mix_suffix,
            )
        )
        order[f"{prefix}.ln2"] = (
            f"{prefix}.ln2.weight",
            f"{prefix}.ln2.bias",
        )
        order[f"{prefix}.ffn"] = (
            f"{prefix}.ffn.x_k",
            f"{prefix}.ffn.key.weight",
            f"{prefix}.ffn.value.weight",
        )
    order["ln_out"] = ("ln_out.weight", "ln_out.bias")
    order["head"] = ("head.weight",)
    return order


def parameter_layout(config: ArchitectureConfig | None = None) -> ParameterTreeLayout:
    """Return the deterministic mapping-pytree identity for the static catalog."""

    return ParameterTreeLayout(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        entries=tuple(
            ParameterTreeLayoutEntry(
                logical_path=descriptor.path,
                jax_keypath=tuple(descriptor.path.split(".")),
                shape=descriptor.shape,
                dtype=descriptor.dtype,
                role=descriptor.role,
                region_ids=descriptor.region_ids,
                trainable=descriptor.trainable_by_default,
                exportable=False,
                metadata=dict(descriptor.metadata),
            )
            for descriptor in parameter_catalog(config).parameters
        ),
    )


def parameter_layout_for_vocabulary(vocabulary_size: int) -> ParameterTreeLayout:
    """Derive a layout from materialized RWKV vocabulary-bearing shapes."""

    if (
        not isinstance(vocabulary_size, int)
        or isinstance(vocabulary_size, bool)
        or vocabulary_size <= 0
    ):
        raise ValueError("vocabulary_size must be positive")
    if vocabulary_size == RWKV7_REFERENCE_VOCABULARY_SIZE:
        return parameter_layout()
    # This helper deliberately has no language identity: execution can derive
    # only the vocabulary range from materialized parameter shapes.
    # Build directly to avoid treating a materialized parameter shape as a
    # complete language configuration.
    descriptors = list(parameter_catalog().parameters)
    for index, descriptor in enumerate(descriptors):
        if descriptor.path == "emb.weight":
            descriptors[index] = ParameterDescriptor(
                path="emb.weight",
                shape=(vocabulary_size, RWKV7_REFERENCE_HIDDEN_SIZE),
                dtype=RWKV7_REFERENCE_DTYPE,
                role="embedding",
                region_ids=("whole_student", "embedding"),
                metadata={"representation": "direct_pinned_numpy_array"},
            )
        if descriptor.path == "head.weight":
            descriptors[index] = ParameterDescriptor(
                path="head.weight",
                shape=(vocabulary_size, RWKV7_REFERENCE_HIDDEN_SIZE),
                dtype=RWKV7_REFERENCE_DTYPE,
                role="output_head",
                region_ids=("whole_student", "output"),
                metadata={"representation": "direct_pinned_numpy_array"},
            )
    catalog = ParameterCatalog(RWKV7_REFERENCE_ARCHITECTURE_ID, tuple(descriptors))
    return ParameterTreeLayout(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
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
                metadata=dict(item.metadata),
            )
            for item in catalog.parameters
        ),
    )


def initialization_parameter_slots(
    config: ArchitectureConfig | None = None,
) -> tuple[str, ...]:
    """Return fixed architecture-owned slots for deterministic initialization."""

    return parameter_catalog(config).paths


def carry_descriptor(config: ArchitectureConfig | None = None) -> dict[str, object]:
    """Describe persistent recurrence state without materializing it in P4.2."""

    shape = equation_shape(config or reference_architecture_config())
    return {
        "schema_version": "radjax.rwkv7_reference_carry.v1",
        "persistent_leaves": {
            "last_x_time": {
                "dtype": RWKV7_REFERENCE_DTYPE,
                "shape": [shape.layer_count, shape.hidden_size],
            },
            "last_x_channel": {
                "dtype": RWKV7_REFERENCE_DTYPE,
                "shape": [shape.layer_count, shape.hidden_size],
            },
            "time_state_matrix": {
                "dtype": RWKV7_REFERENCE_DTYPE,
                "shape": [
                    shape.layer_count,
                    shape.head_count,
                    shape.head_size,
                    shape.head_size,
                ],
            },
        },
        "nonpersistent_token_local_values": ["v0"],
    }


def capability_profile() -> ArchitectureCapabilityProfile:
    return ArchitectureCapabilityProfile(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        capabilities=EXECUTION_CAPABILITIES,
        non_capabilities=(),
        metadata={"phase": "P4.4", "jax_forward_available": True},
    )


def architecture_metadata() -> ArchitectureMetadata:
    catalog = parameter_catalog()
    regions = [NamedRegion("whole_student", catalog.trainable_paths)]
    for block in range(RWKV7_REFERENCE_LAYER_COUNT):
        regions.append(
            NamedRegion(
                f"block_{block}",
                tuple(
                    path
                    for path in catalog.paths
                    if path.startswith(f"blocks.{block}.")
                ),
            )
        )
    regions.extend(
        (
            NamedRegion("embedding", ("emb.weight",)),
            NamedRegion("head", ("head.weight",)),
            NamedRegion("output", ("ln_out.weight", "ln_out.bias", "head.weight")),
        )
    )
    return ArchitectureMetadata(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        parameter_catalog=catalog,
        capability_profile=capability_profile(),
        named_regions=tuple(regions),
        objective_surfaces=(
            IntermediateSurfaceDescriptor(
                surface_id="final_output",
                kind="logits",
                shape_contract={"rank": 3, "shape": ["B", "T", "V"]},
                available_in_training=True,
                available_in_inference=True,
            ),
        ),
        warnings=(
            ArchitectureIssue(
                code="rwkv7_reference_fixture_domain_only",
                message=(
                    "P4.4 JAX execution is proven only against the pinned NumPy "
                    "inference equations on the frozen tiny float32 fixture domain."
                ),
            ),
        ),
        claims_not_made=(
            "equation_parity_outside_fixture_domain_not_claimed",
            "initialization_parity_not_claimed",
            "weight_file_compatibility_not_claimed",
        ),
    )


def hf_descriptor(config: ArchitectureConfig) -> HFCompatibilityDescriptor:
    """Project static architecture identity without claiming HF conversion."""

    validate_rwkv7_config(config)
    shape = equation_shape(config)
    catalog = parameter_catalog(config)
    layout = parameter_layout(config)
    identities = language_identities(config)
    if identities is None:
        tokenizer = HFTokenizerIdentity(
            "rwkv7_reference_fixture_tokenizer",
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
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        architecture_plugin_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        model_type="rwkv7_reference",
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
            "rwkv7_reference_config",
            "rwkv7_reference",
            shape.hidden_size,
            shape.layer_count,
            vocabulary.vocabulary_size,
            config.sequence_length,
            dict(config.model_config),
        ),
        non_claims=(
            "from_pretrained_not_implemented",
            "hf_conversion_not_implemented",
            "save_pretrained_not_implemented",
            "weight_file_compatibility_not_claimed",
        ),
        notes=(
            "P4.4 JAX execution descriptor; no HF conversion or weight-file support."
        ),
    )


__all__ = [
    "STATIC_CAPABILITIES",
    "INITIALIZATION_CAPABILITIES",
    "EXECUTION_CAPABILITIES",
    "CARRY_PYTREE_DESCRIPTOR_DIGEST",
    "architecture_metadata",
    "capability_profile",
    "carry_descriptor",
    "hf_descriptor",
    "initialization_parameter_slots",
    "parameter_catalog",
    "parameter_layout",
    "parameter_layout_for_vocabulary",
    "pinned_numpy_parameter_order",
]
