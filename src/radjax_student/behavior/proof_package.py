"""Closed, deterministic P5.9 HF-shaped behavior-compilation proof package.

This is deliberately a proof envelope, not a Transformers export or a
pretrained-loading interface.  It only serializes verified identities and the
selected neutral configuration as canonical UTF-8 JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from radjax_student.architecture.models import ArchitectureConfig
from radjax_student.artifacts.native_v3_v6 import (
    NativeV3V6BehavioralProjection,
    _require_admitted_native_v3_v6_projection,
)
from radjax_student.behavior.corridor_pass import CorridorCheckpointV1
from radjax_student.behavior.evaluation import HeldOutEvaluationReportV1
from radjax_student.behavior.exemplar_pass import ExemplarCheckpointV1
from radjax_student.contracts.hf import canonical_hf_json
from radjax_student.hf.language_projection import (
    HF_LANGUAGE_PROJECTION_V1,
    HFLanguageProjectionV1,
)

HF_SHAPED_PROOF_PACKAGE_V1 = "hf_shaped_proof_package_v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_NONCLAIMS = (
    "not_a_general_hf_export",
    "not_pretrained_loading",
    "not_phase_6",
    "not_model_quality_or_generalization_evidence",
    "not_an_artifact_transport_or_locator",
)


class HfShapedProofPackageError(ValueError):
    """P5.9 proof-package evidence is incomplete, altered, or mismatched."""


def architecture_config_identity_v1(config: ArchitectureConfig) -> str:
    """Return the canonical identity required by the P5.6--P5.8 lineage."""

    if not isinstance(config, ArchitectureConfig):
        raise HfShapedProofPackageError("selected architecture config is required")
    return _digest(config.to_dict())


def hf_language_projection_identity_v1(projection: HFLanguageProjectionV1) -> str:
    """Return the identity of the neutral P5.3 language projection."""

    if not isinstance(projection, HFLanguageProjectionV1):
        raise HfShapedProofPackageError("HF language projection is required")
    return _digest(_language_payload(projection))


@dataclass(frozen=True)
class HfShapedProofPackageV1:
    """Canonical files plus their complete deterministic inventory."""

    files: tuple[tuple[str, bytes], ...]
    inventory: tuple[tuple[str, str, int], ...]
    package_identity: str
    schema_version: str = HF_SHAPED_PROOF_PACKAGE_V1

    def __post_init__(self) -> None:
        if self.schema_version != HF_SHAPED_PROOF_PACKAGE_V1:
            raise HfShapedProofPackageError("unsupported proof package schema")
        names = tuple(name for name, _ in self.files)
        if not names or names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise HfShapedProofPackageError("proof package inventory names are invalid")
        if any(
            not _safe_file_name(name) or not isinstance(data, bytes)
            for name, data in self.files
        ):
            raise HfShapedProofPackageError("proof package file is invalid")
        expected = tuple((name, _sha256(data), len(data)) for name, data in self.files)
        if self.inventory != expected:
            raise HfShapedProofPackageError("proof package inventory hash mismatch")
        if self.package_identity != _digest(self._identity_payload()):
            raise HfShapedProofPackageError("proof package identity mismatch")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inventory": [list(item) for item in self.inventory],
        }

    def file_bytes(self, name: str) -> bytes:
        """Return one inventory-covered file; no filesystem is involved."""

        for file_name, data in self.files:
            if file_name == name:
                return data
        raise HfShapedProofPackageError("proof package file is absent")

    def to_bytes(self) -> bytes:
        """Encode the complete package deterministically for byte replay."""

        return canonical_hf_json(
            {
                "schema_version": self.schema_version,
                "package_identity": self.package_identity,
                "inventory": [list(item) for item in self.inventory],
                "files": {name: data.decode("utf-8") for name, data in self.files},
            }
        )


def build_hf_shaped_proof_package_v1(
    *,
    projection: NativeV3V6BehavioralProjection,
    architecture_config: ArchitectureConfig,
    corridor_checkpoint: CorridorCheckpointV1,
    final_checkpoint: ExemplarCheckpointV1,
    evaluation_report: HeldOutEvaluationReportV1,
) -> HfShapedProofPackageV1:
    """Build the only P5.9 package from factory-attested P5.3--P5.8 evidence."""

    if not isinstance(projection, NativeV3V6BehavioralProjection):
        raise HfShapedProofPackageError("P5.9 requires a P5.3 projection")
    try:
        _require_admitted_native_v3_v6_projection(projection)
    except ValueError as exc:
        raise HfShapedProofPackageError(
            "P5.3 projection lacks factory provenance"
        ) from exc
    if not isinstance(corridor_checkpoint, CorridorCheckpointV1) or not isinstance(
        final_checkpoint, ExemplarCheckpointV1
    ):
        raise HfShapedProofPackageError("P5.9 requires P5.6 and P5.7 checkpoints")
    if not isinstance(evaluation_report, HeldOutEvaluationReportV1):
        raise HfShapedProofPackageError("P5.9 requires a P5.8 evaluation report")
    _validate_lineage(
        projection,
        architecture_config,
        corridor_checkpoint,
        final_checkpoint,
        evaluation_report,
    )
    files = _files(
        projection,
        architecture_config,
        corridor_checkpoint,
        final_checkpoint,
        evaluation_report,
    )
    inventory = tuple((name, _sha256(data), len(data)) for name, data in files)
    return HfShapedProofPackageV1(
        files=files,
        inventory=inventory,
        package_identity=_digest(
            {
                "schema_version": HF_SHAPED_PROOF_PACKAGE_V1,
                "inventory": [list(item) for item in inventory],
            }
        ),
    )


def replay_hf_shaped_proof_package_v1(
    *, expected: HfShapedProofPackageV1, **kwargs: Any
) -> HfShapedProofPackageV1:
    """Rebuild and require both canonical bytes and package identity to match."""

    if not isinstance(expected, HfShapedProofPackageV1):
        raise HfShapedProofPackageError("expected proof package is required")
    actual = build_hf_shaped_proof_package_v1(**kwargs)
    if (
        actual.package_identity != expected.package_identity
        or actual.to_bytes() != expected.to_bytes()
    ):
        raise HfShapedProofPackageError("proof package replay mismatch")
    return actual


def load_hf_shaped_proof_package_v1(
    data: bytes, *, expected_identity: str
) -> HfShapedProofPackageV1:
    """Decode canonical bytes and reject altered inventory or lineage identity."""

    if not isinstance(data, bytes) or not _SHA256.fullmatch(expected_identity):
        raise HfShapedProofPackageError(
            "proof package bytes or expected identity is invalid"
        )
    try:
        payload = json.loads(data.decode("utf-8"))
        raw_files = payload["files"]
        files = tuple(
            (name, value.encode("utf-8")) for name, value in sorted(raw_files.items())
        )
        package = HfShapedProofPackageV1(
            files=files,
            inventory=tuple(
                (str(name), str(digest), int(size))
                for name, digest, size in payload["inventory"]
            ),
            package_identity=payload["package_identity"],
            schema_version=payload["schema_version"],
        )
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise HfShapedProofPackageError("proof package bytes are invalid") from exc
    if package.to_bytes() != data or package.package_identity != expected_identity:
        raise HfShapedProofPackageError(
            "proof package bytes are altered or lineage mismatched"
        )
    return package


def _validate_lineage(
    projection: NativeV3V6BehavioralProjection,
    config: ArchitectureConfig,
    corridor: CorridorCheckpointV1,
    final: ExemplarCheckpointV1,
    report: HeldOutEvaluationReportV1,
) -> None:
    binding = final.binding
    if corridor.identity != binding.predecessor_checkpoint_identity:
        raise HfShapedProofPackageError("P5.6 to P5.7 checkpoint lineage mismatch")
    if (
        report.binding.final_checkpoint_identity != final.identity
        or report.binding.predecessor_checkpoint_identity != corridor.identity
    ):
        raise HfShapedProofPackageError("P5.8 checkpoint lineage mismatch")
    report_expected = {
        "contract_commit": binding.contract_commit,
        "tome_commit": binding.tome_commit,
        "accepted_receipt_identity": binding.accepted_receipt_identity,
        "language_binding_digest": binding.language_binding_digest,
        "hf_projection_identity": binding.hf_projection_identity,
        "behavioral_source_identity": binding.behavioral_source_identity,
        "behavioral_authority_digest": binding.behavioral_authority_digest,
        "architecture_config_identity": binding.architecture_config_identity,
        "split_identity": binding.split_identity,
        "materialization_identity": binding.materialization_identity,
        "canonical_passport_registry_identity": (
            binding.canonical_passport_registry_identity
        ),
        "corridor_objective_policy_id": binding.corridor_objective_policy_id,
        "reduction_id": binding.reduction_id,
        "corridor_ordering_policy_id": binding.corridor_ordering_policy_id,
        "corridor_batching_policy_id": binding.corridor_batching_policy_id,
        "exemplar_objective_policy_id": binding.exemplar_objective_policy_id,
        "exemplar_ordering_policy_id": binding.ordering_policy_id,
        "exemplar_batching_policy_id": binding.batching_policy_id,
    }
    if any(
        getattr(report.binding, key) != value for key, value in report_expected.items()
    ):
        raise HfShapedProofPackageError("P5.8 authority or policy lineage mismatch")
    expected = {
        "language_binding_digest": projection.language.canonical_binding_digest,
        "hf_projection_identity": hf_language_projection_identity_v1(
            projection.language
        ),
        "behavioral_source_identity": projection.behavioral_source_identity,
        "behavioral_authority_digest": projection.behavioral_authority_digest,
        "architecture_config_identity": architecture_config_identity_v1(config),
    }
    for key, value in expected.items():
        if getattr(binding, key) != value or getattr(report.binding, key) != value:
            raise HfShapedProofPackageError(f"proof package {key} lineage mismatch")
    if not _COMMIT.fullmatch(binding.contract_commit) or not _COMMIT.fullmatch(
        binding.tome_commit
    ):
        raise HfShapedProofPackageError("proof package upstream commit is invalid")
    if not _SHA256.fullmatch(binding.accepted_receipt_identity):
        raise HfShapedProofPackageError(
            "proof package fixture receipt identity is invalid"
        )


def _files(
    projection: NativeV3V6BehavioralProjection,
    config: ArchitectureConfig,
    corridor: CorridorCheckpointV1,
    final: ExemplarCheckpointV1,
    report: HeldOutEvaluationReportV1,
) -> tuple[tuple[str, bytes], ...]:
    binding = final.binding
    payloads = {
        "config.json": {
            "architectures": [config.architecture_id],
            "radjax_architecture_config": config.to_dict(),
        },
        "tokenizer_identity.json": projection.language.tokenizer.to_dict(),
        "vocabulary_identity.json": projection.language.vocabulary.to_dict(),
        "special_tokens_identity.json": projection.language.special_tokens.to_dict(),
        "provenance.json": {
            "hf_language_projection_version": HF_LANGUAGE_PROJECTION_V1,
            "language_binding_digest": projection.language.canonical_binding_digest,
            "hf_projection_identity": hf_language_projection_identity_v1(
                projection.language
            ),
            "contract_commit": binding.contract_commit,
            "tome_commit": binding.tome_commit,
            "fixture_receipt_identity": binding.accepted_receipt_identity,
            "behavioral_source_identity": projection.behavioral_source_identity,
            "behavioral_authority_digest": projection.behavioral_authority_digest,
            "package_semantic_identity": projection.package_semantic_identity,
            "composition_digest": projection.composition_digest,
        },
        "lineage.json": {
            "corridor_checkpoint_identity": corridor.identity,
            "final_checkpoint_identity": final.identity,
            "split_identity": binding.split_identity,
            "materialization_identity": binding.materialization_identity,
            "canonical_passport_registry_identity": (
                binding.canonical_passport_registry_identity
            ),
            "corridor_policy": binding.corridor_objective_policy_id,
            "reduction": binding.reduction_id,
            "exemplar_policy": binding.exemplar_objective_policy_id,
        },
        "optimizer_identity.json": {
            "optimizer_id": final.optimizer_state.envelope.optimizer_id,
            "optimizer_descriptor": final.optimizer_state.descriptor.to_dict(),
            "parameter_layout": final.parameter_layout.to_dict(),
        },
        "held_out_evaluation.json": {
            "identity": report.identity,
            "binding": report.binding.to_dict(),
            "corridor_coordinates": report.corridor_coordinates,
            "exemplar_passport_keys": report.exemplar_passport_keys,
            "corridor_metrics": report.corridor_metrics,
            "exemplar_metrics": report.exemplar_metrics,
        },
        "nonclaims.json": {"nonclaims": _NONCLAIMS},
    }
    return tuple(
        (name, canonical_hf_json(payload)) for name, payload in sorted(payloads.items())
    )


def _language_payload(value: HFLanguageProjectionV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "canonical_binding_digest": value.canonical_binding_digest,
        "tokenizer": value.tokenizer.to_dict(),
        "vocabulary": value.vocabulary.to_dict(),
        "special_tokens": value.special_tokens.to_dict(),
    }


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_hf_json(value)).hexdigest()


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _safe_file_name(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_]+\.json", value))
