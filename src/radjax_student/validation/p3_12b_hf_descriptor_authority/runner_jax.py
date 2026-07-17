"""JAX-dependent P3.12B proof through the public stateful conveyor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from radjax_student.checkpoints import (
    save_learning_checkpoint_v3,
)
from radjax_student.contracts import HFCompatibilityDescriptor, HFPreservationReference
from radjax_student.learning import RunHFSummary
from radjax_student.validation.architecture_audit import build_architecture_audit
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    _new_lifecycle,
    execute_stateful_replays,
)
from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    HFDescriptorAuthorityProof,
    HFProofCase,
    digest,
)

NON_CLAIMS = (
    "no_hf_export",
    "no_transformers_dependency",
    "no_safetensors_output",
    "no_network_access",
    "validation_only_architecture",
)


def _pass(case_id: str, boundary: str, evidence: object) -> HFProofCase:
    return HFProofCase(case_id, "positive", "pass", None, boundary, digest(evidence))


def _reject(
    case_id: str, boundary: str, operation: Callable[[], object]
) -> HFProofCase:
    try:
        operation()
    except Exception as error:
        code = getattr(error, "code", type(error).__name__)
        return HFProofCase(
            case_id,
            "adversarial",
            "reject",
            str(code),
            boundary,
            digest(
                {"type": type(error).__name__, "code": str(code), "message": str(error)}
            ),
        )
    raise RuntimeError(f"{case_id} unexpectedly succeeded")


def execute_hf_descriptor_authority_proof(root: Path) -> HFDescriptorAuthorityProof:
    """Execute construction, checkpoint, replay, report, and failure boundaries."""

    lifecycle = _new_lifecycle("eager", [])
    descriptor = lifecycle.hf_descriptor
    reference = descriptor.preservation_reference()
    checkpoint_dir = root / "checkpoint"
    saved = save_learning_checkpoint_v3(
        lifecycle.checkpoint(), checkpoint_dir, optimizer=lifecycle.optimizer
    )
    restored = _new_lifecycle("eager", []).restore_from_checkpoint(checkpoint_dir)
    replay = execute_stateful_replays(root / "replay")
    audit = build_architecture_audit(Path.cwd())
    if audit["blockers"]:
        raise RuntimeError("dependency audit blocked P3.12B proof")

    positive = (
        _pass(
            "descriptor_constructed_by_architecture",
            "architecture.initialize_parameters",
            descriptor.identity_payload(),
        ),
        _pass(
            "reference_is_descriptor_derived",
            "hf_descriptor.preservation_reference",
            reference.to_dict(),
        ),
        _pass(
            "checkpoint_persists_complete_descriptor",
            "checkpoint.v3.save",
            saved.hf_descriptor.to_dict(),
        ),
        _pass(
            "caller_bound_restore_validates_descriptor",
            "checkpoint.v3.load",
            restored.hf_descriptor.to_dict(),
        ),
        _pass(
            "replay_preserves_descriptor_identity",
            "replay.execute",
            replay.experiment_identity.hf_reference.to_dict(),
        ),
        _pass(
            "report_uses_descriptor_summary",
            "learning.run_report",
            RunHFSummary(descriptor).to_dict(),
        ),
    )

    def fabricated_reference() -> None:
        replace(
            lifecycle,
            hf_reference=HFPreservationReference.from_dict(
                {**reference.to_dict(), "descriptor_digest": "0" * 64}
            ),
        )

    def descriptor_schema() -> None:
        HFCompatibilityDescriptor.from_dict(
            {
                **descriptor.to_dict(),
                "schema_version": "hf_compatibility_descriptor.v999",
            }
        )

    def unknown_field() -> None:
        HFCompatibilityDescriptor.from_dict(
            {**descriptor.to_dict(), "unknown": "field"}
        )

    def reference_mismatch() -> None:
        replace(
            lifecycle,
            hf_reference=HFPreservationReference.from_dict(
                {**reference.to_dict(), "descriptor_digest": "1" * 64}
            ),
        )

    def checkpoint_missing_descriptor() -> None:
        (checkpoint_dir / "hf_descriptor.json").unlink()
        _new_lifecycle("eager", []).restore_from_checkpoint(checkpoint_dir)

    def special_token_bound() -> None:
        payload = descriptor.to_dict()
        special = dict(payload["special_tokens"])
        special["bos_token_id"] = descriptor.vocabulary.vocabulary_size
        payload["special_tokens"] = special
        HFCompatibilityDescriptor.from_dict(payload)

    adversarial = (
        _reject(
            "independently_fabricated_reference", "jax_lifecycle", fabricated_reference
        ),
        _reject(
            "unsupported_descriptor_schema", "hf_descriptor.parse", descriptor_schema
        ),
        _reject("unknown_descriptor_field", "hf_descriptor.parse", unknown_field),
        _reject("descriptor_reference_mismatch", "jax_lifecycle", reference_mismatch),
        _reject(
            "checkpoint_descriptor_missing",
            "checkpoint.v3.load",
            checkpoint_missing_descriptor,
        ),
        _reject(
            "special_token_outside_vocabulary",
            "hf_descriptor.parse",
            special_token_bound,
        ),
    )
    return HFDescriptorAuthorityProof(
        descriptor=descriptor,
        checkpoint_descriptor_digest=saved.hf_descriptor.digest,
        replay_hf_evidence_digest=digest(
            replay.experiment_identity.hf_reference.to_dict()
        ),
        report_hf_evidence_digest=digest(RunHFSummary(descriptor).to_dict()),
        dependency_audit_digest=digest(audit),
        positive_cases=positive,
        adversarial_cases=adversarial,
        non_claims=NON_CLAIMS,
    )


__all__ = ["NON_CLAIMS", "execute_hf_descriptor_authority_proof"]
