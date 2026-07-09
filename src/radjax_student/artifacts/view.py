from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_contract.tome import (
    TomeCompression,
    TomeCoverPage,
    TomeManifest,
    TomePayloadFormat,
    load_tome_cover_page,
    validate_tome,
)
from radjax_contract.tome.inspection import (
    TomeConsumptionPlan,
    inspect_tome_for_consumption,
)
from radjax_contract.vocab import VocabContract


@dataclass(frozen=True)
class TomePayloadSummary:
    payload_format: TomePayloadFormat
    compression: TomeCompression
    expected_adapter: str | None
    implemented_by_contract: bool
    record_count: int | None
    sequence_length: int | None
    shard_count: int
    shard_paths: tuple[str, ...]


@dataclass(frozen=True)
class TomeInferredDefaults:
    role: str | None
    teacher_id: str | None
    teacher_family: str | None
    teacher_backend: str | None
    tokenizer_id: str | None
    vocab_size: int | None
    adapter_family: str | None
    compression_family: str | None
    requires_reconstruction: bool | None


@dataclass(frozen=True)
class TomeArtifactView:
    artifact_dir: Path
    cover_page: TomeCoverPage
    manifest: TomeManifest
    provenance: dict[str, Any]
    payload_summary: TomePayloadSummary
    vocab_contract: VocabContract | None
    tokenizer_contract: dict[str, Any] | None
    sequence_length: int | None
    record_count: int | None
    payload_format: TomePayloadFormat
    inferred_defaults: TomeInferredDefaults
    warnings: tuple[str, ...] = ()


class TomeArtifactError(ValueError):
    def __init__(self, path: str | Path, blockers: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.blockers = blockers
        super().__init__(
            f"could not open Tome artifact at {self.path}: " + ", ".join(blockers)
        )


def open_tome_artifact(path: str | Path) -> TomeArtifactView:
    artifact_dir = Path(path)
    validation = validate_tome(artifact_dir)
    cover_result = load_tome_cover_page(artifact_dir)
    consumption_plan = inspect_tome_for_consumption(artifact_dir)

    blockers = _collect_blockers(
        validation_blockers=validation.blockers,
        cover_blockers=cover_result.blockers,
        consumption_blockers=consumption_plan.blockers,
    )
    if blockers:
        raise TomeArtifactError(artifact_dir, blockers)
    if validation.manifest is None:
        raise TomeArtifactError(artifact_dir, ("manifest_missing",))
    if cover_result.cover_page is None:
        raise TomeArtifactError(artifact_dir, ("cover_page_missing",))

    manifest = validation.manifest
    cover_page = cover_result.cover_page
    return TomeArtifactView(
        artifact_dir=artifact_dir,
        cover_page=cover_page,
        manifest=manifest,
        provenance=dict(manifest.provenance),
        payload_summary=_payload_summary(manifest, cover_page, consumption_plan),
        vocab_contract=manifest.vocab_contract,
        tokenizer_contract=_tokenizer_contract(manifest.vocab_contract),
        sequence_length=manifest.sequence_length,
        record_count=manifest.record_count,
        payload_format=manifest.payload_format,
        inferred_defaults=_inferred_defaults(manifest, cover_page, consumption_plan),
        warnings=tuple(validation.warnings),
    )


def _collect_blockers(
    *,
    validation_blockers: tuple[str, ...],
    cover_blockers: tuple[str, ...],
    consumption_blockers: tuple[str, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for source in (validation_blockers, cover_blockers, consumption_blockers):
        for blocker in source:
            if blocker not in blockers:
                blockers.append(blocker)
    return tuple(blockers)


def _payload_summary(
    manifest: TomeManifest,
    cover_page: TomeCoverPage,
    consumption_plan: TomeConsumptionPlan,
) -> TomePayloadSummary:
    return TomePayloadSummary(
        payload_format=manifest.payload_format,
        compression=manifest.compression,
        expected_adapter=consumption_plan.adapter_id
        or cover_page.student_consumption.expected_adapter,
        implemented_by_contract=cover_page.student_consumption.implemented_by_contract,
        record_count=manifest.record_count,
        sequence_length=manifest.sequence_length,
        shard_count=manifest.shard_count,
        shard_paths=tuple(shard.path for shard in manifest.shards),
    )


def _tokenizer_contract(vocab_contract: VocabContract | None) -> dict[str, Any] | None:
    if vocab_contract is None:
        return None
    return {
        "tokenizer_id": vocab_contract.tokenizer_id,
        "tokenizer_hash": vocab_contract.tokenizer_hash,
        "model_id": vocab_contract.model_id,
        "model_family": vocab_contract.model_family,
        "special_tokens": dict(vocab_contract.special_tokens),
    }


def _inferred_defaults(
    manifest: TomeManifest,
    cover_page: TomeCoverPage,
    consumption_plan: TomeConsumptionPlan,
) -> TomeInferredDefaults:
    vocab = manifest.vocab_contract
    return TomeInferredDefaults(
        role=manifest.role.value,
        teacher_id=cover_page.teacher.teacher_id,
        teacher_family=cover_page.teacher.teacher_family,
        teacher_backend=cover_page.teacher.backend,
        tokenizer_id=None if vocab is None else vocab.tokenizer_id,
        vocab_size=None if vocab is None else vocab.vocab_size,
        adapter_family=consumption_plan.adapter_id
        or cover_page.student_consumption.expected_adapter,
        compression_family=manifest.compression.family.value,
        requires_reconstruction=manifest.compression.requires_reconstruction,
    )
