"""Literal Section G filesystem experiments.

Every function below mutates a real v3 checkpoint tree before invoking the
public save, load, or deterministic-NPZ boundary.
"""

from __future__ import annotations

import hashlib
import io
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from radjax_student.checkpoints import (
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)
from radjax_student.checkpoints.npz_codec import (
    read_deterministic_npz,
    write_deterministic_npz,
)
from radjax_student.contracts import HFPreservationReference
from radjax_student.optimizers import SgdOptimizer
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_directory_experiment,
    execute_memory_experiment,
    public_boundary,
)
from radjax_student.validation.p3_11_10_gate.implementations.literal_fixtures import (
    checkpoint_payload,
    clone_directory,
    load_valid_checkpoint,
    read_json,
    replace_npz_leaf,
    rewrite_integrity,
    rewrite_npz_members,
    write_json,
    write_valid_checkpoint,
)


def _loader(optimizer, layout):
    @public_boundary("checkpoint_restore_validation")
    def load(directory: Path):
        return load_valid_checkpoint(directory, optimizer, layout)

    return load


def _restore(
    context: GateExecutionContext,
    baseline: Path,
    mutated: Path,
    path: str,
    operation: str,
    summary: dict[str, Any],
    optimizer,
    layout,
    baseline_public_input: Any | None = None,
    mutated_public_input: Any | None = None,
) -> ExperimentExecution:
    load = _loader(optimizer, layout)
    return execute_directory_experiment(
        context,
        baseline_directory=baseline,
        mutated_directory=mutated,
        public_input_kind="learning_checkpoint.v3",
        canonical_path=path,
        operation=operation,
        value_summary=summary,
        public_callable=load,
        baseline_callable=load,
        baseline_public_input=baseline_public_input,
        mutated_public_input=mutated_public_input,
    )


def _copy_valid(context: GateExecutionContext, name: str):
    baseline = context.temporary_root / f"{name}-baseline"
    optimizer, layout = write_valid_checkpoint(baseline)
    mutated = clone_directory(baseline, context.temporary_root / f"{name}-mutated")
    return baseline, mutated, optimizer, layout


def experiment_g_deterministic_v3_writes_and_c_fortran_canonical_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"weight": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    mutated = {
        "weight": np.asfortranarray(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
    }

    @public_boundary("checkpoint_restore_validation")
    def write(tree: dict[str, np.ndarray]) -> dict[str, Any]:
        return write_deterministic_npz(
            context.temporary_root / "canonical-order.npz", tree
        )

    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="npz_mapping",
        canonical_path="weight.memory_order",
        operation="replace_c_contiguous_leaf_with_fortran_contiguous_leaf",
        value_summary={"before_order": "C", "after_order": "F"},
        public_callable=write,
        baseline_callable=write,
    )


def experiment_g_v2_read_compatibility_and_v3_continuation_restore(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "v2-compatibility")
    manifest = read_json(mutated / "manifest.json")
    manifest["metadata"] = {"v2_reader": "available"}
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.json.metadata.v2_reader",
        "record_v2_reader_compatibility_metadata",
        {"v2_reader": "available"},
        optimizer,
        layout,
    )


def experiment_g_populated_destination_overwrite(
    context: GateExecutionContext,
) -> ExperimentExecution:
    empty = context.temporary_root / "empty-destination"
    empty.mkdir()
    populated = context.temporary_root / "populated-destination"
    optimizer, _ = write_valid_checkpoint(populated)

    def save(directory: Path):
        return save_learning_checkpoint_v3(
            checkpoint_payload(optimizer), directory, optimizer=optimizer
        )

    save = public_boundary("checkpoint_restore_validation")(save)

    return execute_directory_experiment(
        context,
        baseline_directory=empty,
        mutated_directory=populated,
        public_input_kind="checkpoint_destination",
        canonical_path="destination",
        operation="populate_existing_destination",
        value_summary={"existing_files": True},
        public_callable=save,
        baseline_callable=lambda directory: None,
    )


def experiment_g_incomplete_staged_checkpoint(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "incomplete-staged")
    (mutated / "learning.json").unlink()
    return _restore(
        context,
        baseline,
        mutated,
        "learning.json",
        "delete_staged_learning_component",
        {"removed": "learning.json"},
        optimizer,
        layout,
    )


def experiment_g_missing_manifest(context: GateExecutionContext) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "missing-manifest")
    (mutated / "manifest.json").unlink()
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.json",
        "delete_manifest",
        {"removed": "manifest.json"},
        optimizer,
        layout,
    )


def experiment_g_extra_unexpected_file(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "extra-file")
    (mutated / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    return _restore(
        context,
        baseline,
        mutated,
        "unexpected.txt",
        "add_unexpected_file",
        {"added": "unexpected.txt"},
        optimizer,
        layout,
    )


def experiment_g_missing_expected_sidecar(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "missing-sidecar")
    (mutated / "parameters.npz").unlink()
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz",
        "delete_parameter_sidecar",
        {"removed": "parameters.npz"},
        optimizer,
        layout,
    )


def experiment_g_sidecar_hash_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "sidecar-hash")
    with (mutated / "parameters.npz").open("ab") as stream:
        stream.write(b"tamper")
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz",
        "append_sidecar_bytes",
        {"appended": 6},
        optimizer,
        layout,
    )


def experiment_g_descriptor_hash_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "descriptor-hash")
    path = mutated / "parameters.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.json",
        "change_descriptor_bytes",
        {"suffix": "space"},
        optimizer,
        layout,
    )


def experiment_g_manifest_hash_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "manifest-hash")
    manifest = read_json(mutated / "manifest.json")
    manifest["integrity"]["manifest_digest"] = "0" * 64
    write_json(mutated / "manifest.json", manifest)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.json.integrity.manifest_digest",
        "replace_manifest_digest",
        {"replacement": "zero_digest"},
        optimizer,
        layout,
    )


def experiment_g_malformed_json_descriptor(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "malformed-descriptor")
    (mutated / "parameters.json").write_text("{", encoding="utf-8")
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.json",
        "write_malformed_json",
        {"bytes": "{"},
        optimizer,
        layout,
    )


def experiment_g_malformed_npz_member_name(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "bad-member")
    rewrite_npz_members(
        mutated / "parameters.npz",
        additions={"../escape.npy": b"not-a-npy"},
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz:../escape.npy",
        "add_malformed_npz_member_name",
        {"member": "../escape.npy"},
        optimizer,
        layout,
    )


def experiment_g_extra_npz_member(context: GateExecutionContext) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "extra-member")
    rewrite_npz_members(
        mutated / "parameters.npz",
        additions={"extra.npy": b"not-a-npy"},
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz:extra.npy",
        "add_extra_npz_member",
        {"member": "extra.npy"},
        optimizer,
        layout,
    )


def experiment_g_missing_npz_member(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "missing-member")
    descriptor = read_json(mutated / "parameters.json")
    first_member = descriptor["leaves"][0]["member"]
    rewrite_npz_members(mutated / "parameters.npz", removals=(first_member,))
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        f"parameters.npz:{first_member}",
        "remove_required_npz_member",
        {"removed": first_member},
        optimizer,
        layout,
    )


def experiment_g_object_dtype(context: GateExecutionContext) -> ExperimentExecution:
    baseline = {"weight": np.asarray([1], dtype=np.int32)}
    # The object dtype is itself the forbidden input.  Use a deterministic
    # object payload so repeated public writer failures have the same message
    # identity rather than inheriting an object-address repr.
    mutated = {"weight": np.asarray(["forbidden"], dtype=object)}

    def write(tree):
        return write_deterministic_npz(context.temporary_root / "object.npz", tree)

    write = public_boundary("checkpoint_restore_validation")(write)
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="npz_mapping",
        canonical_path="weight.dtype",
        operation="replace_numeric_dtype_with_object_dtype",
        value_summary={"dtype": "object"},
        public_callable=write,
        baseline_callable=write,
    )


def experiment_g_structured_dtype(context: GateExecutionContext) -> ExperimentExecution:
    baseline = {"weight": np.asarray([1], dtype=np.int32)}
    mutated = {"weight": np.asarray([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])}

    def write(tree):
        return write_deterministic_npz(context.temporary_root / "structured.npz", tree)

    write = public_boundary("checkpoint_restore_validation")(write)
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="npz_mapping",
        canonical_path="weight.dtype",
        operation="replace_plain_dtype_with_structured_dtype",
        value_summary={"dtype": "structured"},
        public_callable=write,
        baseline_callable=write,
    )


def experiment_g_wrong_tensor_shape(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "wrong-shape")
    descriptor = read_json(mutated / "parameters.json")
    replace_npz_leaf(
        mutated / "parameters.npz",
        descriptor,
        ("trunk", "weight"),
        np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz.trunk.weight.shape",
        "reshape_parameter_leaf",
        {"shape": [2, 1]},
        optimizer,
        layout,
    )


def experiment_g_wrong_tensor_dtype(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "wrong-dtype")
    descriptor = read_json(mutated / "parameters.json")
    replace_npz_leaf(
        mutated / "parameters.npz",
        descriptor,
        ("trunk", "weight"),
        np.asarray([1], dtype=np.int32),
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "parameters.npz.trunk.weight.dtype",
        "cast_parameter_leaf_to_int32",
        {"dtype": "int32"},
        optimizer,
        layout,
    )


def experiment_g_noncanonical_array_order(
    context: GateExecutionContext,
) -> ExperimentExecution:
    canonical = context.temporary_root / "canonical-order.npz"
    descriptor = write_deterministic_npz(
        canonical,
        {"weight": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    )
    baseline = {
        "descriptor": descriptor,
        "sidecar": canonical.read_bytes(),
    }
    member = descriptor["leaves"][0]["member"]
    fortran_stream = io.BytesIO()
    np.lib.format.write_array(
        fortran_stream,
        np.asfortranarray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        version=(1, 0),
        allow_pickle=False,
    )
    rewrite_npz_members(canonical, additions={member: fortran_stream.getvalue()})
    mutated = {
        "descriptor": descriptor,
        "sidecar": canonical.read_bytes(),
    }

    @public_boundary("checkpoint_restore_validation")
    def read(payload: dict[str, Any]) -> Any:
        sidecar = context.temporary_root / "public-noncanonical-order.npz"
        sidecar.write_bytes(payload["sidecar"])
        return read_deterministic_npz(sidecar, payload["descriptor"])

    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="npz_sidecar_bytes",
        canonical_path="weight.fortran_order",
        operation="replace_canonical_npy_member_with_fortran_order_member",
        value_summary={"before_order": "C", "after_order": "F"},
        public_callable=read,
        baseline_callable=read,
    )


def experiment_g_optimizer_id_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "optimizer-id")
    manifest = read_json(mutated / "manifest.json")
    manifest["optimizer"]["optimizer_id"] = "foreign.optimizer"
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.optimizer.optimizer_id",
        "replace_optimizer_identity",
        {"optimizer_id": "foreign.optimizer"},
        optimizer,
        layout,
    )


def experiment_g_optimizer_capability_version_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "optimizer-capability")
    manifest = read_json(mutated / "manifest.json")
    manifest["optimizer"]["optimizer_capability_version"] = 999
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.optimizer.optimizer_capability_version",
        "replace_optimizer_capability",
        {"capability": 999},
        optimizer,
        layout,
    )


def experiment_g_optimizer_schema_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "optimizer-schema")
    manifest = read_json(mutated / "manifest.json")
    manifest["optimizer"]["optimizer_numerical_state_schema_version"] = (
        "foreign.schema.v1"
    )
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.optimizer.optimizer_numerical_state_schema_version",
        "replace_optimizer_schema",
        {"schema": "foreign.schema.v1"},
        optimizer,
        layout,
    )


def experiment_g_optimizer_envelope_step_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "envelope-step")
    payload = read_json(mutated / "optimizer_state.json")
    payload["envelope"]["step"] = 3
    write_json(mutated / "optimizer_state.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "optimizer_state.json.envelope.step",
        "increment_envelope_step_only",
        {"before": 2, "after": 3},
        optimizer,
        layout,
    )


def experiment_g_optimizer_numerical_step_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "numerical-step")
    payload = read_json(mutated / "optimizer_state.json")
    replace_npz_leaf(
        mutated / "optimizer_state.npz",
        payload["numerical_state_descriptor"],
        ("step",),
        np.asarray(3, dtype=np.int32),
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "optimizer_state.npz.step",
        "increment_numerical_step_only",
        {"before": 2, "after": 3},
        optimizer,
        layout,
    )


def experiment_g_optimizer_parameter_paths_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "optimizer-paths")
    payload = read_json(mutated / "optimizer_state.json")
    payload["envelope"]["parameter_paths"] = ["foreign.weight"]
    write_json(mutated / "optimizer_state.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "optimizer_state.json.envelope.parameter_paths",
        "replace_optimizer_parameter_paths",
        {"path": "foreign.weight"},
        optimizer,
        layout,
    )


def experiment_g_parameter_layout_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "layout")
    payload = read_json(mutated / "layout.json")
    payload["architecture_id"] = "foreign.architecture"
    write_json(mutated / "layout.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "layout.json.architecture_id",
        "replace_layout_architecture_identity",
        {"architecture_id": "foreign.architecture"},
        optimizer,
        layout,
    )


def experiment_g_hf_identity_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "hf-identity")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["hf_reference"]["model_type"] = "foreign-model"
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.hf_reference.model_type",
        "replace_hf_model_type",
        {"model_type": "foreign-model"},
        optimizer,
        layout,
    )


def experiment_g_tokenizer_identity_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "tokenizer")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["hf_reference"]["tokenizer_id"] = "foreign-tokenizer"
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.hf_reference.tokenizer_id",
        "replace_tokenizer_identity",
        {"tokenizer_id": "foreign-tokenizer"},
        optimizer,
        layout,
    )


def experiment_g_vocabulary_size_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "vocabulary")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["hf_reference"]["vocabulary_size"] = 999
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.hf_reference.vocabulary_size",
        "replace_vocabulary_size",
        {"vocabulary_size": 999},
        optimizer,
        layout,
    )


def experiment_g_special_token_digest_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "special-token")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["hf_reference"]["special_token_digest"] = "f" * 64
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.hf_reference.special_token_digest",
        "replace_special_token_digest",
        {"digest": "f" * 64},
        optimizer,
        layout,
    )


def experiment_g_architecture_config_digest_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "config-digest")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["architecture_config_digest"] = "f" * 64
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.architecture_config_digest",
        "replace_architecture_config_digest",
        {"digest": "f" * 64},
        optimizer,
        layout,
    )


def experiment_g_parameter_catalog_digest_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "catalog-digest")
    manifest = read_json(mutated / "manifest.json")
    manifest["architecture"]["parameter_catalog_digest"] = "e" * 64
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.architecture.parameter_catalog_digest",
        "replace_parameter_catalog_digest",
        {"digest": "e" * 64},
        optimizer,
        layout,
    )


def experiment_g_architecture_state_identity_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "state-identity")
    payload = read_json(mutated / "learning.json")
    payload["architecture_state"] = {"state_id": "foreign-state"}
    write_json(mutated / "learning.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "learning.json.architecture_state.state_id",
        "replace_architecture_state_identity",
        {"state_id": "foreign-state"},
        optimizer,
        layout,
    )


def experiment_g_architecture_carry_descriptor_tampering(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "carry-descriptor")
    payload = read_json(mutated / "architecture_carry.json")
    payload["schema_version"] = "foreign.carry.v1"
    write_json(mutated / "architecture_carry.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "architecture_carry.json.schema_version",
        "replace_carry_descriptor_schema",
        {"schema": "foreign.carry.v1"},
        optimizer,
        layout,
    )


def experiment_g_carry_sidecar_changed_rehashed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "carry-sidecar")
    descriptor = read_json(mutated / "architecture_carry.json")
    replace_npz_leaf(
        mutated / "architecture_carry.npz",
        descriptor,
        ("count",),
        np.asarray([99], dtype=np.int32),
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "architecture_carry.npz.count",
        "reshape_carry_leaf_and_rehash",
        {"before_shape": [], "after_shape": [1]},
        optimizer,
        layout,
    )


def experiment_g_optimizer_sidecar_changed_rehashed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "optimizer-sidecar")
    payload = read_json(mutated / "optimizer_state.json")
    replace_npz_leaf(
        mutated / "optimizer_state.npz",
        payload["numerical_state_descriptor"],
        ("step",),
        np.asarray(9, dtype=np.int32),
    )
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "optimizer_state.npz.step",
        "replace_optimizer_value_and_rehash",
        {"before": 2, "after": 9},
        optimizer,
        layout,
    )


def experiment_g_runtime_reference_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "runtime-reference")

    def load(directory: Path):
        return load_learning_checkpoint_v3(
            directory,
            optimizer=optimizer,
            parameter_layout=layout,
            runtime_reference="foreign-runtime",
        )

    load = public_boundary("checkpoint_restore_validation")(load)
    return execute_directory_experiment(
        context,
        baseline_directory=baseline,
        mutated_directory=mutated,
        public_input_kind="checkpoint_load_request",
        canonical_path="runtime_reference",
        operation="replace_runtime_reference",
        value_summary={"runtime_reference": "foreign-runtime"},
        public_callable=load,
        baseline_callable=lambda directory: load_valid_checkpoint(
            directory, optimizer, layout
        ),
        baseline_public_input={
            "directory": baseline,
            "runtime_reference": "literal-runtime-reference",
        },
        mutated_public_input={
            "directory": mutated,
            "runtime_reference": "foreign-runtime",
        },
    )


def experiment_g_caller_expected_identity_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "expected-hf")
    checkpoint = load_valid_checkpoint(baseline, optimizer, layout)
    foreign = HFPreservationReference(
        checkpoint.hf_reference.descriptor_schema_version,
        checkpoint.hf_reference.descriptor_digest,
        checkpoint.hf_reference.model_type,
        checkpoint.hf_reference.architecture_id,
        "foreign-tokenizer",
        checkpoint.hf_reference.vocabulary_size,
        checkpoint.hf_reference.special_token_digest,
        checkpoint.hf_reference.parameter_layout_digest,
        checkpoint.hf_reference.architecture_config_digest,
    )

    def load(directory: Path):
        return load_learning_checkpoint_v3(
            directory,
            optimizer=optimizer,
            parameter_layout=layout,
            expected_hf_reference=foreign,
        )

    load = public_boundary("checkpoint_restore_validation")(load)
    return execute_directory_experiment(
        context,
        baseline_directory=baseline,
        mutated_directory=mutated,
        public_input_kind="checkpoint_load_request",
        canonical_path="expected_hf_reference.tokenizer_id",
        operation="replace_expected_tokenizer_identity",
        value_summary={"tokenizer_id": "foreign-tokenizer"},
        public_callable=load,
        baseline_callable=lambda directory: load_valid_checkpoint(
            directory, optimizer, layout
        ),
        baseline_public_input={
            "directory": baseline,
            "expected_hf_reference": checkpoint.hf_reference,
        },
        mutated_public_input={"directory": mutated, "expected_hf_reference": foreign},
    )


def experiment_g_caller_omits_required_lifecycle_expectations(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "missing-expectations")
    checkpoint = load_valid_checkpoint(baseline, optimizer, layout)
    valid_carry = checkpoint.architecture_carry_descriptor
    baseline_request = {
        "directory": baseline,
        "expected_hf_reference": checkpoint.hf_reference,
        "expected_architecture_config_digest": checkpoint.architecture_config_digest,
        "expected_parameter_catalog_digest": checkpoint.parameter_catalog_digest,
        "expected_architecture_carry_descriptor": valid_carry,
    }
    mutated_request = {
        "directory": mutated,
        "expected_hf_reference": None,
        "expected_architecture_config_digest": None,
        "expected_parameter_catalog_digest": None,
        "expected_architecture_carry_descriptor": None,
    }

    @public_boundary("checkpoint_restore_validation")
    def load(request: dict[str, Any]):
        return load_learning_checkpoint_v3(
            request["directory"],
            optimizer=optimizer,
            parameter_layout=layout,
            expected_hf_reference=request["expected_hf_reference"],
            expected_architecture_config_digest=request[
                "expected_architecture_config_digest"
            ],
            expected_parameter_catalog_digest=request[
                "expected_parameter_catalog_digest"
            ],
            expected_architecture_carry_descriptor=request[
                "expected_architecture_carry_descriptor"
            ],
            require_lifecycle_expectations=True,
        )

    return execute_memory_experiment(
        context,
        baseline=baseline_request,
        mutated=mutated_request,
        public_input_kind="checkpoint_load_request",
        canonical_path="expected_lifecycle_identity",
        operation="remove_caller_lifecycle_expectations",
        value_summary={"expected_hf_reference": None},
        public_callable=load,
        baseline_callable=load,
    )


def experiment_g_silent_state_repair(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "silent-repair")
    payload = read_json(mutated / "optimizer_state.json")
    payload["envelope"]["step"] = 5
    write_json(mutated / "optimizer_state.json", payload)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "optimizer_state.json.envelope.step",
        "set_inconsistent_optimizer_step",
        {"before": 2, "after": 5},
        optimizer,
        layout,
    )


def experiment_g_crash_preserves_existing_destination(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = context.temporary_root / "crash-baseline"
    baseline.mkdir()
    mutated = context.temporary_root / "crash-destination"
    optimizer = SgdOptimizer()
    existing = checkpoint_payload(optimizer)
    existing = replace(
        existing,
        parameters={"trunk": {"weight": np.asarray([2.0], dtype=np.float32)}},
    )
    save_learning_checkpoint_v3(existing, mutated, optimizer=optimizer)
    before = hashlib.sha256((mutated / "manifest.json").read_bytes()).hexdigest()

    def save(directory: Path):
        result = save_learning_checkpoint_v3(
            checkpoint_payload(optimizer), directory, optimizer=optimizer
        )
        after = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if after != before:
            raise RuntimeError("existing checkpoint changed during rejected save")
        return result

    save = public_boundary("checkpoint_restore_validation")(save)
    return execute_directory_experiment(
        context,
        baseline_directory=baseline,
        mutated_directory=mutated,
        public_input_kind="checkpoint_destination",
        canonical_path="manifest.json",
        operation="attempt_atomic_overwrite_of_existing_checkpoint",
        value_summary={"existing_manifest_digest": before},
        public_callable=save,
        baseline_callable=lambda directory: None,
    )


def experiment_g_v2_presented_as_v3(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "v2-presented")
    shutil.rmtree(mutated)
    mutated.mkdir()
    (mutated / "learning_checkpoint.v2.json").write_text("{}\n", encoding="utf-8")
    return _restore(
        context,
        baseline,
        mutated,
        "learning_checkpoint.v2.json",
        "replace_v3_tree_with_v2_payload",
        {"schema": "learning_checkpoint.v2"},
        optimizer,
        layout,
    )


def experiment_g_unsupported_future_schema(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline, mutated, optimizer, layout = _copy_valid(context, "future-schema")
    manifest = read_json(mutated / "manifest.json")
    manifest["schema_version"] = "learning_checkpoint.v99"
    write_json(mutated / "manifest.json", manifest)
    rewrite_integrity(mutated)
    return _restore(
        context,
        baseline,
        mutated,
        "manifest.json.schema_version",
        "replace_schema_with_future_version",
        {"schema_version": "learning_checkpoint.v99"},
        optimizer,
        layout,
    )


SECTION_IMPLEMENTATIONS = {
    "G.positive.deterministic_v3_writes_and_c_fortran_canonical_identity": GateCaseImplementation(  # noqa: E501
        experiment_g_deterministic_v3_writes_and_c_fortran_canonical_identity
    ),
    "G.positive.v2_read_compatibility_and_v3_continuation_restore": GateCaseImplementation(  # noqa: E501
        experiment_g_v2_read_compatibility_and_v3_continuation_restore
    ),
    "G.reject.populated_destination_overwrite": GateCaseImplementation(
        experiment_g_populated_destination_overwrite
    ),
    "G.reject.incomplete_staged_checkpoint": GateCaseImplementation(
        experiment_g_incomplete_staged_checkpoint
    ),
    "G.reject.missing_manifest": GateCaseImplementation(experiment_g_missing_manifest),
    "G.reject.extra_unexpected_file": GateCaseImplementation(
        experiment_g_extra_unexpected_file
    ),
    "G.reject.missing_expected_sidecar": GateCaseImplementation(
        experiment_g_missing_expected_sidecar
    ),
    "G.reject.sidecar_hash_mismatch": GateCaseImplementation(
        experiment_g_sidecar_hash_mismatch
    ),
    "G.reject.descriptor_hash_mismatch": GateCaseImplementation(
        experiment_g_descriptor_hash_mismatch
    ),
    "G.reject.manifest_hash_mismatch": GateCaseImplementation(
        experiment_g_manifest_hash_mismatch
    ),
    "G.reject.malformed_json_descriptor": GateCaseImplementation(
        experiment_g_malformed_json_descriptor
    ),
    "G.reject.malformed_npz_member_name": GateCaseImplementation(
        experiment_g_malformed_npz_member_name
    ),
    "G.reject.extra_npz_member": GateCaseImplementation(experiment_g_extra_npz_member),
    "G.reject.missing_npz_member": GateCaseImplementation(
        experiment_g_missing_npz_member
    ),
    "G.reject.object_dtype": GateCaseImplementation(experiment_g_object_dtype),
    "G.reject.structured_dtype": GateCaseImplementation(experiment_g_structured_dtype),
    "G.reject.wrong_tensor_shape": GateCaseImplementation(
        experiment_g_wrong_tensor_shape
    ),
    "G.reject.wrong_tensor_dtype": GateCaseImplementation(
        experiment_g_wrong_tensor_dtype
    ),
    "G.reject.noncanonical_array_order": GateCaseImplementation(
        experiment_g_noncanonical_array_order
    ),
    "G.reject.optimizer_id_tampering": GateCaseImplementation(
        experiment_g_optimizer_id_tampering
    ),
    "G.reject.optimizer_capability_version_tampering": GateCaseImplementation(
        experiment_g_optimizer_capability_version_tampering
    ),
    "G.reject.optimizer_schema_tampering": GateCaseImplementation(
        experiment_g_optimizer_schema_tampering
    ),
    "G.reject.optimizer_envelope_step_tampering": GateCaseImplementation(
        experiment_g_optimizer_envelope_step_tampering
    ),
    "G.reject.optimizer_numerical_step_tampering": GateCaseImplementation(
        experiment_g_optimizer_numerical_step_tampering
    ),
    "G.reject.optimizer_parameter_paths_tampering": GateCaseImplementation(
        experiment_g_optimizer_parameter_paths_tampering
    ),
    "G.reject.parameter_layout_tampering": GateCaseImplementation(
        experiment_g_parameter_layout_tampering
    ),
    "G.reject.hf_identity_tampering": GateCaseImplementation(
        experiment_g_hf_identity_tampering
    ),
    "G.reject.tokenizer_identity_tampering": GateCaseImplementation(
        experiment_g_tokenizer_identity_tampering
    ),
    "G.reject.vocabulary_size_tampering": GateCaseImplementation(
        experiment_g_vocabulary_size_tampering
    ),
    "G.reject.special_token_digest_tampering": GateCaseImplementation(
        experiment_g_special_token_digest_tampering
    ),
    "G.reject.architecture_config_digest_tampering": GateCaseImplementation(
        experiment_g_architecture_config_digest_tampering
    ),
    "G.reject.parameter_catalog_digest_tampering": GateCaseImplementation(
        experiment_g_parameter_catalog_digest_tampering
    ),
    "G.reject.architecture_state_identity_tampering": GateCaseImplementation(
        experiment_g_architecture_state_identity_tampering
    ),
    "G.reject.architecture_carry_descriptor_tampering": GateCaseImplementation(
        experiment_g_architecture_carry_descriptor_tampering
    ),
    "G.reject.carry_sidecar_changed_rehashed": GateCaseImplementation(
        experiment_g_carry_sidecar_changed_rehashed
    ),
    "G.reject.optimizer_sidecar_changed_rehashed": GateCaseImplementation(
        experiment_g_optimizer_sidecar_changed_rehashed
    ),
    "G.reject.runtime_reference_mismatch": GateCaseImplementation(
        experiment_g_runtime_reference_mismatch
    ),
    "G.reject.caller_expected_identity_mismatch": GateCaseImplementation(
        experiment_g_caller_expected_identity_mismatch
    ),
    "G.reject.caller_omits_required_lifecycle_expectations": GateCaseImplementation(
        experiment_g_caller_omits_required_lifecycle_expectations
    ),
    "G.reject.silent_state_repair": GateCaseImplementation(
        experiment_g_silent_state_repair
    ),
    "G.reject.crash_preserves_existing_destination": GateCaseImplementation(
        experiment_g_crash_preserves_existing_destination
    ),
    "G.reject.v2_presented_as_v3": GateCaseImplementation(
        experiment_g_v2_presented_as_v3
    ),
    "G.reject.unsupported_future_schema": GateCaseImplementation(
        experiment_g_unsupported_future_schema
    ),
}


__all__ = ["SECTION_IMPLEMENTATIONS"]
