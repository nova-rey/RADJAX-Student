from __future__ import annotations

from pathlib import Path

import numpy as np
from radjax_contract.io import write_json, write_jsonl
from radjax_contract.tome import (
    TomeBehavioralSummary,
    TomeCompression,
    TomeContentsSummary,
    TomeCorpusSource,
    TomeCorpusSummary,
    TomeCoverPage,
    TomeManifest,
    TomePayloadFormat,
    TomeRole,
    TomeSplitSummary,
    TomeStudentConsumptionSummary,
    TomeTeacherSummary,
)
from radjax_contract.tome.inspection import expected_adapter_for_payload
from radjax_contract.vocab import VocabContract


def write_dense_tome(
    path: Path,
    *,
    records: list[dict[str, object]] | None = None,
    logits: np.ndarray | None = None,
    vocab_size: int = 3,
    sequence_length: int = 2,
    manifest: TomeManifest | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = records or [
        {"example_id": "example-0", "text": "alpha"},
        {"example_id": "example-1", "text": "beta"},
    ]
    values = (
        logits
        if logits is not None
        else np.ones((len(rows), sequence_length, vocab_size), dtype=np.float32)
    )
    write_jsonl(path / "records.jsonl", rows)
    np.save(path / "logits.npy", values, allow_pickle=False)
    tome_manifest = manifest or TomeManifest(
        vocab_contract=VocabContract(tokenizer_id="toy", vocab_size=vocab_size),
        record_count=len(rows),
        sequence_length=sequence_length,
    )
    write_json(path / "manifest.json", tome_manifest.to_dict())
    write_json(
        path / "cover_page.json",
        _cover_page(
            record_count=len(rows),
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            dtype=str(values.dtype),
        ).to_dict(),
    )
    return path


def _cover_page(
    *,
    record_count: int,
    sequence_length: int,
    vocab_size: int,
    dtype: str,
) -> TomeCoverPage:
    payload_format = TomePayloadFormat.DENSE_LOGITS_V0
    compression = TomeCompression()
    return TomeCoverPage(
        title="Tiny dense logits smoke Tome",
        description="Dense teacher-output Tome generated from tiny examples.",
        teacher=TomeTeacherSummary(
            teacher_id="fake-teacher",
            teacher_family="fake",
            backend="fake",
            teacher_dtype=dtype,
            teacher_vocab_size=vocab_size,
        ),
        corpus=TomeCorpusSummary(
            summary="Tiny synthetic smoke corpus.",
            sources=(
                TomeCorpusSource(
                    source_id="synthetic_smoke",
                    source_type="synthetic",
                    description="Small checked test fixture corpus.",
                    record_count=record_count,
                ),
            ),
            contains_synthetic_examples=True,
        ),
        contents=TomeContentsSummary(
            role=TomeRole.TRAINING,
            record_count=record_count,
            sequence_length=sequence_length,
            payload_format=payload_format,
            compression=compression,
        ),
        behavioral_fingerprint=TomeBehavioralSummary(),
        splits=TomeSplitSummary(split_role=TomeRole.TRAINING),
        student_consumption=TomeStudentConsumptionSummary(
            expected_adapter=expected_adapter_for_payload(payload_format),
            implemented_by_contract=True,
            notes="Student may consume directly without reconstruction.",
        ),
    )
