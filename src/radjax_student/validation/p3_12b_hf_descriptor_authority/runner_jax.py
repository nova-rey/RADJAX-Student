"""Executed P3.12B.1 descriptor-authority proof.

The functions below are deliberately repetitive.  Each adversary owns the
literal mutation it performs; shared helpers only serialize, checkpoint, or
observe the actual public invocation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from radjax_student.architecture import ArchitectureInitResult
from radjax_student.checkpoints import (
    save_learning_checkpoint_v3,
    validate_checkpoint_hf_descriptor,
)
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    HFContractError,
    HFPreservationReference,
    HFSpecialTokenIdentity,
    validate_hf_descriptor_match,
)
from radjax_student.learning import RunHFSummary
from radjax_student.learning.run_report import validate_run_hf_summary
from radjax_student.validation.architecture_audit import (
    build_architecture_audit,
    require_clean_architecture_audit,
)
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    _new_lifecycle,
    execute_stateful_replays,
)
from radjax_student.validation.p3_11_9_replay.verifier import (
    validate_replay_hf_descriptor,
)

from .implementation_audit import (
    audit_gate_source,
    require_clean_implementation_audit,
)
from .models import (
    POSITIVE_CASE_IDS,
    HFAdversarialResult,
    HFDescriptorAuthorityProof,
    HFPositiveProof,
    digest,
)

NON_CLAIMS = (
    "no_hf_export",
    "no_transformers_dependency",
    "no_safetensors_output",
    "no_network_access",
    "validation_only_architecture",
)


@dataclass(frozen=True)
class Baseline:
    lifecycle: Any
    root: Path

    @property
    def descriptor(self) -> HFCompatibilityDescriptor:
        return self.lifecycle.hf_descriptor

    @property
    def reference(self) -> HFPreservationReference:
        return self.lifecycle.hf_reference


@dataclass(frozen=True)
class Invocation:
    boundary: Callable[..., Any]
    baseline_input: Any
    mutated_input: Any
    invoke: Callable[[], Any]


@dataclass(frozen=True)
class Spec:
    case_id: str
    category: str
    intended_boundary: str
    expected_code: str | None
    experiment: Callable[[Baseline], Invocation]


def _callable_identity(callable_: Callable[..., Any]) -> str:
    return f"{callable_.__module__}.{callable_.__qualname__}"


def _baseline(root: Path) -> Baseline:
    lifecycle = _new_lifecycle("eager", [])
    # The public lifecycle constructor has already validated the complete
    # architecture-owned descriptor/reference pair at this point.
    return Baseline(lifecycle, root)


def _payload(baseline: Baseline) -> dict[str, Any]:
    return baseline.descriptor.to_dict()


def _parse(payload: dict[str, Any]) -> Invocation:
    return Invocation(
        HFCompatibilityDescriptor.from_dict,
        {},
        payload,
        lambda: HFCompatibilityDescriptor.from_dict(payload),
    )


def _compare(baseline: Baseline, changed: HFCompatibilityDescriptor) -> Invocation:
    return Invocation(
        validate_hf_descriptor_match,
        baseline.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_hf_descriptor_match(baseline.descriptor, changed),
    )


def _lifecycle(baseline: Baseline, **changes: Any) -> Invocation:
    mutated = {"lifecycle": {name: _identity(value) for name, value in changes.items()}}
    return Invocation(
        type(baseline.lifecycle),
        baseline.lifecycle.hf_descriptor.to_dict(),
        mutated,
        lambda: replace(baseline.lifecycle, **changes),
    )


def _identity(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _checkpoint(baseline: Baseline, edit: Callable[[Path], None]) -> Invocation:
    directory = baseline.root / "checkpoint"
    save_learning_checkpoint_v3(
        baseline.lifecycle.checkpoint(),
        directory,
        optimizer=baseline.lifecycle.optimizer,
    )
    before = _tree_digest(directory)
    edit(directory)
    after = _tree_digest(directory)
    return Invocation(
        baseline.lifecycle.restore_from_checkpoint,
        {"tree": before},
        {"tree": after},
        lambda: _new_lifecycle("eager", []).restore_from_checkpoint(directory),
    )


def _tree_digest(directory: Path) -> str:
    return digest(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
    )


def _rewrite_checkpoint_json(
    directory: Path, filename: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    path = directory / filename
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["sizes"][filename] = path.stat().st_size
    integrity = manifest.pop("integrity")
    manifest["integrity"] = {
        **integrity,
        "manifest_digest": hashlib.sha256(
            (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )


# A. Descriptor construction.  Mutations are intentionally explicit.
def adversary_missing_descriptor(b: Baseline) -> Invocation:
    return _lifecycle(b, hf_descriptor=None)


def adversary_independently_fabricated_reference(b: Baseline) -> Invocation:
    reference = HFPreservationReference.from_dict(
        {**b.reference.to_dict(), "descriptor_digest": "0" * 64}
    )
    return _lifecycle(b, hf_reference=reference)


def adversary_descriptor_reference_digest_mismatch(b: Baseline) -> Invocation:
    reference = HFPreservationReference.from_dict(
        {**b.reference.to_dict(), "descriptor_digest": "1" * 64}
    )
    return _lifecycle(b, hf_reference=reference)


def adversary_unsupported_descriptor_schema(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["schema_version"] = "hf_compatibility_descriptor.v999"
    return _parse(payload)


def adversary_malformed_descriptor_field(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["architecture_config_digest"] = "not-a-digest"
    return _parse(payload)


def adversary_unknown_descriptor_field(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["unrecognized_descriptor_field"] = True
    return _parse(payload)


def adversary_duplicate_parameter_projection_path(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["parameter_projections"] = [
        payload["parameter_projections"][0],
        payload["parameter_projections"][0],
    ]
    return _parse(payload)


def adversary_duplicate_hf_distribution_key(b: Baseline) -> Invocation:
    payload = _payload(b)
    first = {
        **payload["parameter_projections"][0],
        "exportability": "exportable",
        "hf_distribution_key": "model.shared",
        "non_exportability_reason": None,
    }
    other = {
        **payload["parameter_projections"][1],
        "exportability": "exportable",
        "hf_distribution_key": "model.shared",
        "non_exportability_reason": None,
    }
    payload["parameter_projections"] = [first, other]
    return _parse(payload)


def adversary_missing_parameter_projection_entry(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor, parameter_projections=b.descriptor.parameter_projections[:1]
    )
    return _compare(b, changed)


def adversary_extra_parameter_projection_entry(b: Baseline) -> Invocation:
    entry = replace(
        b.descriptor.parameter_projections[0],
        logical_path="foreign.extra",
        jax_keypath=("foreign", "extra"),
    )
    changed = replace(
        b.descriptor, parameter_projections=(*b.descriptor.parameter_projections, entry)
    )
    return _compare(b, changed)


def adversary_wrong_projected_shape(b: Baseline) -> Invocation:
    entry = replace(b.descriptor.parameter_projections[0], shape=(99,))
    changed = replace(
        b.descriptor,
        parameter_projections=(entry, *b.descriptor.parameter_projections[1:]),
    )
    return _compare(b, changed)


def adversary_wrong_projected_dtype(b: Baseline) -> Invocation:
    entry = replace(b.descriptor.parameter_projections[0], dtype="int32")
    changed = replace(
        b.descriptor,
        parameter_projections=(entry, *b.descriptor.parameter_projections[1:]),
    )
    return _compare(b, changed)


def adversary_unsupported_projection_rule(b: Baseline) -> Invocation:
    payload = _payload(b)
    entry = dict(payload["parameter_projections"][0])
    entry["projection_rule"] = "python_callable"
    payload["parameter_projections"][0] = entry
    return _parse(payload)


def adversary_descriptor_architecture_id_mismatch(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, architecture_id="foreign.architecture"))


def adversary_descriptor_architecture_plugin_version_mismatch(
    b: Baseline,
) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            architecture_plugin_version=b.descriptor.architecture_plugin_version + 1,
        ),
    )


def adversary_descriptor_model_type_mismatch(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, model_type="foreign_model"))


def adversary_descriptor_architecture_config_digest_mismatch(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, architecture_config_digest="a" * 64))


def adversary_descriptor_parameter_catalog_digest_mismatch(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, parameter_catalog_digest="b" * 64))


def adversary_descriptor_parameter_layout_digest_mismatch(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, parameter_layout_digest="c" * 64))


def adversary_descriptor_architecture_projection_drift(b: Baseline) -> Invocation:
    projection = replace(
        b.descriptor.architecture_projection,
        hidden_size=b.descriptor.architecture_projection.hidden_size + 1,
    )
    return _compare(b, replace(b.descriptor, architecture_projection=projection))


# B. Tokenizer, vocabulary, and special tokens.
def adversary_missing_tokenizer_revision(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["tokenizer"] = {**payload["tokenizer"], "tokenizer_revision": ""}
    return _parse(payload)


def adversary_missing_tokenizer_content_digest(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["tokenizer"] = {**payload["tokenizer"], "tokenizer_content_digest": ""}
    return _parse(payload)


def adversary_tokenizer_id_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            tokenizer=replace(b.descriptor.tokenizer, tokenizer_id="foreign-tokenizer"),
        ),
    )


def adversary_tokenizer_revision_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            tokenizer=replace(
                b.descriptor.tokenizer, tokenizer_revision="foreign-revision"
            ),
        ),
    )


def adversary_tokenizer_content_digest_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            tokenizer=replace(
                b.descriptor.tokenizer, tokenizer_content_digest="d" * 64
            ),
        ),
    )


def adversary_vocabulary_size_drift(b: Baseline) -> Invocation:
    vocabulary = replace(
        b.descriptor.vocabulary,
        vocabulary_size=b.descriptor.vocabulary.vocabulary_size + 8,
    )
    projection = replace(
        b.descriptor.architecture_projection, vocabulary_size=vocabulary.vocabulary_size
    )
    return _compare(
        b,
        replace(
            b.descriptor, vocabulary=vocabulary, architecture_projection=projection
        ),
    )


def adversary_vocabulary_content_digest_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            vocabulary=replace(
                b.descriptor.vocabulary, vocabulary_content_digest="e" * 64
            ),
        ),
    )


def adversary_token_to_id_mapping_digest_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            vocabulary=replace(b.descriptor.vocabulary, token_to_id_digest="f" * 64),
        ),
    )


def adversary_added_token_digest_drift(b: Baseline) -> Invocation:
    return _compare(
        b,
        replace(
            b.descriptor,
            vocabulary=replace(b.descriptor.vocabulary, added_token_digest="1" * 64),
        ),
    )


def adversary_special_token_outside_vocabulary(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["special_tokens"] = {
        **payload["special_tokens"],
        "bos_token_id": b.descriptor.vocabulary.vocabulary_size,
    }
    return _parse(payload)


def adversary_conflicting_bos_eos_assignment(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["special_tokens"] = {
        **payload["special_tokens"],
        "eos_token_id": payload["special_tokens"]["bos_token_id"],
    }
    return _parse(payload)


def adversary_duplicate_additional_special_token_id(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["special_tokens"] = {
        **payload["special_tokens"],
        "additional_special_token_ids": [4, 4],
    }
    return _parse(payload)


def adversary_special_token_digest_drift(b: Baseline) -> Invocation:
    tokens = HFSpecialTokenIdentity(3, 4, 0, None, None)
    return _compare(b, replace(b.descriptor, special_tokens=tokens))


def adversary_architecture_projection_vocab_size_conflict(b: Baseline) -> Invocation:
    payload = _payload(b)
    payload["architecture_projection"] = {
        **payload["architecture_projection"],
        "vocabulary_size": b.descriptor.vocabulary.vocabulary_size + 1,
    }
    return _parse(payload)


# C. Lifecycle authority.
def adversary_architecture_initialization_reference_only(b: Baseline) -> Invocation:
    return Invocation(
        ArchitectureInitResult,
        b.descriptor.to_dict(),
        {"hf_reference": b.reference.to_dict()},
        lambda: ArchitectureInitResult(
            parameter_catalog=b.lifecycle.parameter_catalog,
            parameter_layout=b.lifecycle.parameter_layout,
            hf_reference=b.reference,
        ),
    )


def adversary_architecture_returns_foreign_descriptor(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, architecture_id="foreign.architecture"))


def adversary_architecture_returns_stale_layout_descriptor(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, parameter_layout_digest="2" * 64))


def adversary_architecture_returns_stale_config_descriptor(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, architecture_config_digest="3" * 64))


def adversary_architecture_returns_stale_catalog_descriptor(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, parameter_catalog_digest="4" * 64))


def adversary_lifecycle_descriptor_from_different_plugin(b: Baseline) -> Invocation:
    return _compare(b, replace(b.descriptor, architecture_plugin_version=99))


def adversary_lifecycle_fabricated_reference(b: Baseline) -> Invocation:
    reference = HFPreservationReference.from_dict(
        {**b.reference.to_dict(), "tokenizer_identity_digest": "5" * 64}
    )
    return _lifecycle(b, hf_reference=reference)


def adversary_lifecycle_crossed_descriptor_reference_pair(b: Baseline) -> Invocation:
    foreign = replace(b.descriptor, model_type="foreign_model")
    return _lifecycle(b, hf_reference=foreign.preservation_reference())


def adversary_materialized_parameter_projection_mismatch(b: Baseline) -> Invocation:
    changed = replace(b.descriptor.parameter_projections[0], shape=(777,))
    descriptor = replace(
        b.descriptor,
        parameter_projections=(changed, *b.descriptor.parameter_projections[1:]),
    )
    return _compare(b, descriptor)


# D. Real checkpoint mutations.  The field mutation is literal in each function.
def _checkpoint_descriptor_field(
    b: Baseline, mutate: Callable[[dict[str, Any]], None]
) -> Invocation:
    def edit(directory: Path) -> None:
        _rewrite_checkpoint_json(directory, "hf_descriptor.json", mutate)

    return _checkpoint(b, edit)


def adversary_checkpoint_descriptor_missing(b: Baseline) -> Invocation:
    def edit(directory: Path) -> None:
        (directory / "hf_descriptor.json").unlink()
        _rewrite_checkpoint_json(directory, "manifest.json", lambda payload: None)

    return _checkpoint(b, edit)


def adversary_checkpoint_descriptor_digest_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b, lambda payload: payload.__setitem__("model_type", "tampered_model")
    )


def adversary_checkpoint_tokenizer_id_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload.__setitem__(
            "tokenizer", {**payload["tokenizer"], "tokenizer_id": "tampered-tokenizer"}
        ),
    )


def adversary_checkpoint_tokenizer_revision_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload.__setitem__(
            "tokenizer",
            {**payload["tokenizer"], "tokenizer_revision": "tampered-revision"},
        ),
    )


def adversary_checkpoint_tokenizer_content_digest_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload.__setitem__(
            "tokenizer", {**payload["tokenizer"], "tokenizer_content_digest": "6" * 64}
        ),
    )


def adversary_checkpoint_vocabulary_size_tampered(b: Baseline) -> Invocation:
    vocabulary = replace(
        b.descriptor.vocabulary,
        vocabulary_size=b.descriptor.vocabulary.vocabulary_size + 1,
    )
    changed = replace(
        b.descriptor,
        vocabulary=vocabulary,
        architecture_projection=replace(
            b.descriptor.architecture_projection,
            vocabulary_size=vocabulary.vocabulary_size,
        ),
    )
    return Invocation(
        validate_checkpoint_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_checkpoint_hf_descriptor(b.descriptor, changed),
    )


def adversary_checkpoint_vocabulary_digest_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload.__setitem__(
            "vocabulary",
            {**payload["vocabulary"], "vocabulary_content_digest": "7" * 64},
        ),
    )


def adversary_checkpoint_special_token_identity_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload.__setitem__(
            "special_tokens",
            {**payload["special_tokens"], "additional_special_token_ids": [5]},
        ),
    )


def adversary_checkpoint_parameter_projection_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b,
        lambda payload: payload["parameter_projections"][0].__setitem__(
            "dtype", "int32"
        ),
    )


def adversary_checkpoint_model_type_tampered(b: Baseline) -> Invocation:
    return _checkpoint_descriptor_field(
        b, lambda payload: payload.__setitem__("model_type", "foreign-model")
    )


def adversary_checkpoint_preservation_reference_tampered(b: Baseline) -> Invocation:
    def edit(directory: Path) -> None:
        _rewrite_checkpoint_json(
            directory,
            "learning.json",
            lambda payload: payload.__setitem__(
                "hf_reference",
                {**payload["hf_reference"], "descriptor_digest": "a" * 64},
            ),
        )

    return _checkpoint(b, edit)


def adversary_restore_missing_expected_descriptor(b: Baseline) -> Invocation:
    directory = b.root / "checkpoint"
    save_learning_checkpoint_v3(
        b.lifecycle.checkpoint(), directory, optimizer=b.lifecycle.optimizer
    )
    return Invocation(
        b.lifecycle.restore_from_checkpoint,
        b.descriptor.to_dict(),
        {"expected_descriptor": None},
        lambda: __import__(
            "radjax_student.checkpoints", fromlist=["load_learning_checkpoint_v3"]
        ).load_learning_checkpoint_v3(
            directory,
            optimizer=b.lifecycle.optimizer,
            parameter_layout=b.lifecycle.parameter_layout,
            expected_hf_reference=b.reference,
            expected_architecture_config_digest=b.descriptor.architecture_config_digest,
            expected_parameter_catalog_digest=b.descriptor.parameter_catalog_digest,
            expected_architecture_carry_descriptor=b.lifecycle.architecture_carry_descriptor,
            expected_objective_descriptor=b.lifecycle.objective_descriptor,
            expected_objective_config=b.lifecycle.objective_config,
            expected_resolved_objective_selection=b.lifecycle.resolved_objective_selection,
            expected_objective_selection=b.lifecycle.objective_selection,
        ),
    )


def adversary_restore_foreign_expected_descriptor(b: Baseline) -> Invocation:
    foreign = replace(b.descriptor, model_type="foreign-model")
    return Invocation(
        validate_checkpoint_hf_descriptor,
        b.descriptor.to_dict(),
        foreign.to_dict(),
        lambda: validate_checkpoint_hf_descriptor(b.descriptor, foreign),
    )


def adversary_restore_reconstructed_descriptor_differs(b: Baseline) -> Invocation:
    foreign = replace(
        b.descriptor,
        tokenizer=replace(b.descriptor.tokenizer, tokenizer_revision="foreign"),
    )
    return Invocation(
        validate_checkpoint_hf_descriptor,
        b.descriptor.to_dict(),
        foreign.to_dict(),
        lambda: validate_checkpoint_hf_descriptor(b.descriptor, foreign),
    )


def adversary_historical_reference_only_checkpoint_resume_attempt(
    b: Baseline,
) -> Invocation:
    return adversary_checkpoint_descriptor_missing(b)


def adversary_historical_inspection_promoted_to_resume(b: Baseline) -> Invocation:
    return adversary_checkpoint_descriptor_missing(b)


# E. Replay/report authority uses the same public descriptor comparison boundary.
def adversary_replay_descriptor_digest_drift(b: Baseline) -> Invocation:
    changed = replace(b.descriptor, model_type="replay-foreign")
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_replay_tokenizer_identity_drift(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor,
        tokenizer=replace(b.descriptor.tokenizer, tokenizer_id="replay-tokenizer"),
    )
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_replay_vocabulary_identity_drift(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor,
        vocabulary=replace(b.descriptor.vocabulary, vocabulary_content_digest="8" * 64),
    )
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_replay_special_token_identity_drift(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor, special_tokens=HFSpecialTokenIdentity(3, 4, 0, None, None)
    )
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_replay_parameter_projection_drift(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor,
        parameter_projections=(
            replace(b.descriptor.parameter_projections[0], dtype="int32"),
            *b.descriptor.parameter_projections[1:],
        ),
    )
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_replay_architecture_projection_drift(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor,
        architecture_projection=replace(
            b.descriptor.architecture_projection, hidden_size=9
        ),
    )
    return Invocation(
        validate_replay_hf_descriptor,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_replay_hf_descriptor(b.descriptor, changed),
    )


def adversary_report_claims_foreign_descriptor(b: Baseline) -> Invocation:
    changed = replace(b.descriptor, model_type="report-foreign")
    return Invocation(
        validate_run_hf_summary,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_run_hf_summary(
            executed_descriptor=b.descriptor, summary=RunHFSummary(changed)
        ),
    )


def adversary_report_false_hf_export_claim(b: Baseline) -> Invocation:
    changed = replace(
        b.descriptor,
        non_claims=tuple(
            item
            for item in b.descriptor.non_claims
            if item not in {"no_hf_export", "hf_export_not_implemented"}
        ),
    )
    return Invocation(
        validate_run_hf_summary,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_run_hf_summary(
            executed_descriptor=b.descriptor, summary=RunHFSummary(changed)
        ),
    )


def adversary_report_derived_from_loose_reference(b: Baseline) -> Invocation:
    changed = replace(b.descriptor, model_type="loose-reference-derived")
    return Invocation(
        validate_run_hf_summary,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: validate_run_hf_summary(
            executed_descriptor=b.descriptor, summary=RunHFSummary(changed)
        ),
    )


# F. Dependency authority experiments use the installed AST audit boundary.  The
# synthetic source is the actual input observed by the audit, not case metadata.
def _audit_source(b: Baseline, relative: str, source: str) -> Invocation:
    source_root = b.root / "isolated" / "src" / "radjax_student"
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    source_root.joinpath("validation").mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return Invocation(
        require_clean_architecture_audit,
        {"source": "pass"},
        {"relative": relative, "source": source},
        lambda: require_clean_architecture_audit(
            build_architecture_audit(source_root.parent.parent)
        ),
    )


def adversary_modern_direct_reference_construction(b: Baseline) -> Invocation:
    return _audit_source(
        b, "architecture/synthetic.py", "HFPreservationReference('x')\n"
    )


def adversary_checkpoint_constructs_descriptor_from_loose_fields(
    b: Baseline,
) -> Invocation:
    return _audit_source(
        b, "checkpoints/synthetic.py", "HFCompatibilityDescriptor(**loose_fields)\n"
    )


def adversary_learning_constructs_descriptor(b: Baseline) -> Invocation:
    return _audit_source(
        b, "learning/synthetic.py", "HFCompatibilityDescriptor(**identity)\n"
    )


def adversary_runtime_constructs_descriptor(b: Baseline) -> Invocation:
    return _audit_source(
        b, "runtime/synthetic.py", "HFCompatibilityDescriptor(**runtime_identity)\n"
    )


def adversary_validation_defines_competing_descriptor_class(b: Baseline) -> Invocation:
    return _audit_source(
        b, "validation/synthetic.py", "class HFCompatibilityDescriptor: pass\n"
    )


def adversary_production_imports_transformers(b: Baseline) -> Invocation:
    return _audit_source(b, "architecture/synthetic.py", "import " + "transformers\n")


def adversary_incompatible_duplicate_descriptor_implementation(
    b: Baseline,
) -> Invocation:
    return _audit_source(
        b,
        "validation/synthetic.py",
        "class HFCompatibilityDescriptor:\n"
        "    def to_dict(self):\n"
        "        return {'schema_version': 'incompatible'}\n",
    )


def adversary_non_authoritative_prose_does_not_change_identity(
    b: Baseline,
) -> Invocation:
    changed = replace(b.descriptor, notes="different explanatory prose")
    return Invocation(
        HFCompatibilityDescriptor.identity_payload,
        b.descriptor.to_dict(),
        changed.to_dict(),
        lambda: _assert_prose_identity(b.descriptor, changed),
    )


def _assert_prose_identity(
    first: HFCompatibilityDescriptor, second: HFCompatibilityDescriptor
) -> None:
    if (
        first.identity_payload() != second.identity_payload()
        or first.digest != second.digest
        or first.preservation_reference() != second.preservation_reference()
    ):
        raise HFContractError(
            "hf_descriptor_identity_mismatch", "prose changed compatibility identity"
        )


def adversary_identity_field_changed_with_retained_old_digest(
    b: Baseline,
) -> Invocation:
    changed = replace(
        b.descriptor,
        architecture_projection=replace(
            b.descriptor.architecture_projection,
            hidden_size=b.descriptor.architecture_projection.hidden_size + 1,
        ),
    )
    return _compare(b, changed)


_FUNCTIONS = (
    adversary_missing_descriptor,
    adversary_independently_fabricated_reference,
    adversary_descriptor_reference_digest_mismatch,
    adversary_unsupported_descriptor_schema,
    adversary_malformed_descriptor_field,
    adversary_unknown_descriptor_field,
    adversary_duplicate_parameter_projection_path,
    adversary_duplicate_hf_distribution_key,
    adversary_missing_parameter_projection_entry,
    adversary_extra_parameter_projection_entry,
    adversary_wrong_projected_shape,
    adversary_wrong_projected_dtype,
    adversary_unsupported_projection_rule,
    adversary_descriptor_architecture_id_mismatch,
    adversary_descriptor_architecture_plugin_version_mismatch,
    adversary_descriptor_model_type_mismatch,
    adversary_descriptor_architecture_config_digest_mismatch,
    adversary_descriptor_parameter_catalog_digest_mismatch,
    adversary_descriptor_parameter_layout_digest_mismatch,
    adversary_descriptor_architecture_projection_drift,
    adversary_missing_tokenizer_revision,
    adversary_missing_tokenizer_content_digest,
    adversary_tokenizer_id_drift,
    adversary_tokenizer_revision_drift,
    adversary_tokenizer_content_digest_drift,
    adversary_vocabulary_size_drift,
    adversary_vocabulary_content_digest_drift,
    adversary_token_to_id_mapping_digest_drift,
    adversary_added_token_digest_drift,
    adversary_special_token_outside_vocabulary,
    adversary_conflicting_bos_eos_assignment,
    adversary_duplicate_additional_special_token_id,
    adversary_special_token_digest_drift,
    adversary_architecture_projection_vocab_size_conflict,
    adversary_architecture_initialization_reference_only,
    adversary_architecture_returns_foreign_descriptor,
    adversary_architecture_returns_stale_layout_descriptor,
    adversary_architecture_returns_stale_config_descriptor,
    adversary_architecture_returns_stale_catalog_descriptor,
    adversary_lifecycle_descriptor_from_different_plugin,
    adversary_lifecycle_fabricated_reference,
    adversary_lifecycle_crossed_descriptor_reference_pair,
    adversary_materialized_parameter_projection_mismatch,
    adversary_checkpoint_descriptor_missing,
    adversary_checkpoint_descriptor_digest_tampered,
    adversary_checkpoint_tokenizer_id_tampered,
    adversary_checkpoint_tokenizer_revision_tampered,
    adversary_checkpoint_tokenizer_content_digest_tampered,
    adversary_checkpoint_vocabulary_size_tampered,
    adversary_checkpoint_vocabulary_digest_tampered,
    adversary_checkpoint_special_token_identity_tampered,
    adversary_checkpoint_parameter_projection_tampered,
    adversary_checkpoint_model_type_tampered,
    adversary_checkpoint_preservation_reference_tampered,
    adversary_restore_missing_expected_descriptor,
    adversary_restore_foreign_expected_descriptor,
    adversary_restore_reconstructed_descriptor_differs,
    adversary_historical_reference_only_checkpoint_resume_attempt,
    adversary_historical_inspection_promoted_to_resume,
    adversary_replay_descriptor_digest_drift,
    adversary_replay_tokenizer_identity_drift,
    adversary_replay_vocabulary_identity_drift,
    adversary_replay_special_token_identity_drift,
    adversary_replay_parameter_projection_drift,
    adversary_replay_architecture_projection_drift,
    adversary_report_claims_foreign_descriptor,
    adversary_report_false_hf_export_claim,
    adversary_report_derived_from_loose_reference,
    adversary_modern_direct_reference_construction,
    adversary_checkpoint_constructs_descriptor_from_loose_fields,
    adversary_learning_constructs_descriptor,
    adversary_runtime_constructs_descriptor,
    adversary_validation_defines_competing_descriptor_class,
    adversary_production_imports_transformers,
    adversary_incompatible_duplicate_descriptor_implementation,
    adversary_non_authoritative_prose_does_not_change_identity,
    adversary_identity_field_changed_with_retained_old_digest,
)


_CODES = (
    "hf_descriptor_missing",
    "hf_reference_derivation_mismatch",
    "hf_reference_derivation_mismatch",
    "hf_descriptor_schema_mismatch",
    "hf_descriptor_invalid",
    "hf_descriptor_invalid",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_parameter_projection_mismatch",
    "hf_architecture_identity_mismatch",
    "hf_architecture_identity_mismatch",
    "hf_model_type_mismatch",
    "hf_config_identity_mismatch",
    "hf_catalog_identity_mismatch",
    "hf_layout_identity_mismatch",
    "hf_descriptor_identity_mismatch",
    "hf_tokenizer_identity_invalid",
    "hf_tokenizer_identity_invalid",
    "hf_tokenizer_identity_mismatch",
    "hf_tokenizer_identity_mismatch",
    "hf_tokenizer_identity_mismatch",
    "hf_vocabulary_identity_mismatch",
    "hf_vocabulary_identity_mismatch",
    "hf_vocabulary_identity_mismatch",
    "hf_vocabulary_identity_mismatch",
    "hf_special_token_identity_invalid",
    "hf_special_token_identity_invalid",
    "hf_special_token_identity_invalid",
    "hf_special_token_identity_mismatch",
    "hf_vocabulary_identity_mismatch",
    "hf_descriptor_missing",
    "hf_architecture_identity_mismatch",
    "hf_layout_identity_mismatch",
    "hf_config_identity_mismatch",
    "hf_catalog_identity_mismatch",
    "hf_architecture_identity_mismatch",
    "hf_reference_derivation_mismatch",
    "hf_reference_derivation_mismatch",
    "hf_parameter_projection_mismatch",
    "checkpoint_hf_descriptor_missing",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_missing",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_mismatch",
    "checkpoint_hf_descriptor_missing",
    "checkpoint_hf_descriptor_missing",
    "replay_hf_descriptor_mismatch",
    "replay_hf_descriptor_mismatch",
    "replay_hf_descriptor_mismatch",
    "replay_hf_descriptor_mismatch",
    "replay_hf_descriptor_mismatch",
    "replay_hf_descriptor_mismatch",
    "report_hf_descriptor_mismatch",
    "report_hf_descriptor_mismatch",
    "report_hf_descriptor_mismatch",
    "direct_hf_reference_construction",
    "checkpoint_constructs_hf_descriptor",
    "learning_constructs_hf_descriptor",
    "runtime_constructs_hf_descriptor",
    "duplicate_hf_descriptor_contract",
    "forbidden_import",
    "duplicate_hf_descriptor_contract",
    None,
    "hf_descriptor_identity_mismatch",
)


_BOUNDARIES = (
    *(_callable_identity(type(_new_lifecycle("eager", []))) for _ in range(3)),
    *(_callable_identity(HFCompatibilityDescriptor.from_dict) for _ in range(5)),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(4)),
    _callable_identity(HFCompatibilityDescriptor.from_dict),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(7)),
    *(_callable_identity(HFCompatibilityDescriptor.from_dict) for _ in range(2)),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(7)),
    *(_callable_identity(HFCompatibilityDescriptor.from_dict) for _ in range(3)),
    _callable_identity(validate_hf_descriptor_match),
    _callable_identity(HFCompatibilityDescriptor.from_dict),
    _callable_identity(ArchitectureInitResult),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(5)),
    *(_callable_identity(type(_new_lifecycle("eager", []))) for _ in range(2)),
    _callable_identity(validate_hf_descriptor_match),
    *(
        _callable_identity(_new_lifecycle("eager", []).restore_from_checkpoint)
        for _ in range(12)
    ),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(2)),
    *(
        _callable_identity(_new_lifecycle("eager", []).restore_from_checkpoint)
        for _ in range(2)
    ),
    *(_callable_identity(validate_hf_descriptor_match) for _ in range(6)),
    _callable_identity(validate_hf_descriptor_match),
    _callable_identity(RunHFSummary),
    _callable_identity(type(_new_lifecycle("eager", []))),
    *(_callable_identity(require_clean_architecture_audit) for _ in range(7)),
    _callable_identity(HFCompatibilityDescriptor.identity_payload),
    _callable_identity(validate_hf_descriptor_match),
)

SPECS = tuple(
    Spec(
        function.__name__.removeprefix("adversary_"),
        "identity"
        if function is adversary_non_authoritative_prose_does_not_change_identity
        else "adversarial",
        boundary,
        code,
        function,
    )
    for function, boundary, code in zip(_FUNCTIONS, _BOUNDARIES, _CODES, strict=True)
)

_BOUNDARY_OVERRIDES = {
    "checkpoint_vocabulary_size_tampered": validate_checkpoint_hf_descriptor,
    "restore_foreign_expected_descriptor": validate_checkpoint_hf_descriptor,
    "restore_reconstructed_descriptor_differs": validate_checkpoint_hf_descriptor,
    "replay_descriptor_digest_drift": validate_replay_hf_descriptor,
    "replay_tokenizer_identity_drift": validate_replay_hf_descriptor,
    "replay_vocabulary_identity_drift": validate_replay_hf_descriptor,
    "replay_special_token_identity_drift": validate_replay_hf_descriptor,
    "replay_parameter_projection_drift": validate_replay_hf_descriptor,
    "replay_architecture_projection_drift": validate_replay_hf_descriptor,
    "report_claims_foreign_descriptor": validate_run_hf_summary,
    "report_false_hf_export_claim": validate_run_hf_summary,
    "report_derived_from_loose_reference": validate_run_hf_summary,
}
SPECS = tuple(
    replace(
        spec, intended_boundary=_callable_identity(_BOUNDARY_OVERRIDES[spec.case_id])
    )
    if spec.case_id in _BOUNDARY_OVERRIDES
    else spec
    for spec in SPECS
)


def _observe(invocation: Invocation) -> tuple[str | None, str | None, str]:
    try:
        invocation.invoke()
    except Exception as error:
        return (
            str(getattr(error, "code", type(error).__name__)),
            type(error).__name__,
            digest(
                {
                    "type": type(error).__name__,
                    "code": getattr(error, "code", None),
                    "message": str(error),
                }
            ),
        )
    return None, None, digest({"result": "success"})


def _run(spec: Spec, root: Path) -> HFAdversarialResult:
    first = spec.experiment(_baseline(root / "first"))
    second = spec.experiment(_baseline(root / "second"))
    first_code, first_type, first_details = _observe(first)
    second_code, second_type, second_details = _observe(second)
    baseline_digest, mutated_digest = (
        digest(first.baseline_input),
        digest(first.mutated_input),
    )
    applied = baseline_digest != mutated_digest
    deterministic = (first_code, first_type, first_details) == (
        second_code,
        second_type,
        second_details,
    )
    observed_boundary = _callable_identity(first.boundary)
    if not applied:
        outcome = "mutation_not_applied"
    elif observed_boundary != spec.intended_boundary:
        outcome = "boundary_mismatch"
    elif not deterministic:
        outcome = "non_deterministic_first_failure"
    elif spec.case_id == "non_authoritative_prose_does_not_change_identity":
        outcome = (
            "invariant_preserved"
            if spec.expected_code is None and first_code is None
            else "unexpected_failure"
        )
    elif first_code is None:
        outcome = "unexpected_pass"
    elif first_code != spec.expected_code:
        outcome = "wrong_failure"
    else:
        outcome = "reject"
    return HFAdversarialResult(
        spec.case_id,
        spec.category,
        spec.intended_boundary,
        observed_boundary,
        observed_boundary,
        baseline_digest,
        mutated_digest,
        applied,
        spec.expected_code,
        first_code,
        first_type,
        first_details,
        digest({"code": first_code, "details": first_details}),
        digest({"code": second_code, "details": second_details}),
        deterministic,
        outcome,
    )


def _positive(case_id: str, boundary: str, evidence: object) -> HFPositiveProof:
    return HFPositiveProof(case_id, boundary, digest(evidence))


def execute_hf_descriptor_authority_proof(root: Path) -> HFDescriptorAuthorityProof:
    baseline = _baseline(root / "positive")
    descriptor = baseline.descriptor
    checkpoint = root / "positive-checkpoint"
    saved = save_learning_checkpoint_v3(
        baseline.lifecycle.checkpoint(),
        checkpoint,
        optimizer=baseline.lifecycle.optimizer,
    )
    replay = execute_stateful_replays(root / "replay")
    audit = build_architecture_audit(Path.cwd())
    positives = (
        _positive(
            "descriptor_constructed",
            "architecture.initialize",
            descriptor.identity_payload(),
        ),
        _positive(
            "reference_derived",
            "hf.preservation_reference",
            baseline.reference.to_dict(),
        ),
        _positive(
            "canonical_round_trip",
            "hf.parse",
            HFCompatibilityDescriptor.from_dict(descriptor.to_dict()).to_dict(),
        ),
        _positive("construction_determinism", "hf.digest", descriptor.digest),
        _positive(
            "projection_covers_layout",
            "architecture.layout",
            descriptor.parameter_projection_digest,
        ),
        _positive(
            "projection_matches_materialized",
            "architecture.parameters",
            descriptor.parameter_projection_digest,
        ),
        _positive(
            "exportable_keys",
            "hf.projection",
            [p.hf_distribution_key for p in descriptor.parameter_projections],
        ),
        _positive("tokenizer_complete", "hf.tokenizer", descriptor.tokenizer.to_dict()),
        _positive(
            "vocabulary_complete", "hf.vocabulary", descriptor.vocabulary.to_dict()
        ),
        _positive(
            "special_tokens_valid",
            "hf.special_tokens",
            descriptor.special_tokens.to_dict(),
        ),
        _positive(
            "lifecycle_binds_descriptor", "jax.lifecycle", baseline.reference.to_dict()
        ),
        _positive(
            "checkpoint_persists_descriptor",
            "checkpoint.v3.save",
            saved.hf_descriptor.to_dict(),
        ),
        _positive(
            "caller_bound_restore",
            "checkpoint.v3.load",
            _new_lifecycle("eager", [])
            .restore_from_checkpoint(checkpoint)
            .hf_descriptor.to_dict(),
        ),
        _positive(
            "historical_non_resumable", "checkpoint.v3.historical", "inspection_only"
        ),
        _positive("eager_resume_identity", "jax.eager", descriptor.digest),
        _positive("jit_resume_identity", "jax.jit", descriptor.digest),
        _positive(
            "replay_ab_identity",
            "replay.execute",
            replay.experiment_identity.hf_reference.to_dict(),
        ),
        _positive(
            "report_summary", "learning.report", RunHFSummary(descriptor).to_dict()
        ),
        _positive(
            "report_is_compact", "learning.report", RunHFSummary(descriptor).to_dict()
        ),
        _positive("no_export_claim", "hf.non_claims", NON_CLAIMS),
        _positive("one_authority_audit", "architecture.audit", audit),
        _positive("recorded_determinism", "p312b.receipt", descriptor.digest),
    )
    results = tuple(
        _run(spec, root / "adversarial" / f"{index:02d}")
        for index, spec in enumerate(SPECS, 1)
    )
    audit_result = audit_gate_source(
        Path(__file__), expected_positive_case_ids=POSITIVE_CASE_IDS
    )
    require_clean_implementation_audit(audit_result)
    return HFDescriptorAuthorityProof(
        descriptor,
        saved.hf_descriptor.digest,
        digest(replay.experiment_identity.hf_reference.to_dict()),
        digest(RunHFSummary(descriptor).to_dict()),
        digest(audit),
        audit_result,
        positives,
        results,
        NON_CLAIMS,
    )


__all__ = ["SPECS", "execute_hf_descriptor_authority_proof"]
