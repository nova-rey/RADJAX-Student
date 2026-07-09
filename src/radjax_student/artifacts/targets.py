from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from radjax_contract.io import read_json
from radjax_contract.tome import (
    TomeManifest,
    TomePayloadFormat,
    TomeRecord,
    load_tome_records,
)
from radjax_contract.tome.payloads import DEFAULT_DENSE_LOGITS_PAYLOAD
from radjax_contract.validation import validate_teacher_tome


@dataclass(frozen=True)
class DenseTomeTargets:
    artifact_dir: Path
    manifest: TomeManifest
    records: tuple[TomeRecord, ...]
    logits: np.ndarray

    def probabilities(self, *, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        scaled = self.logits.astype(np.float64) / float(temperature)
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


def load_dense_tome_targets(path: str | Path) -> DenseTomeTargets:
    artifact_dir = Path(path)
    validation = validate_teacher_tome(artifact_dir)
    if not validation.ok:
        raise ValueError(
            "teacher tome is not loadable: " + ", ".join(validation.blockers)
        )

    manifest = TomeManifest.from_dict(read_json(artifact_dir / "manifest.json"))
    if manifest.payload_format is not TomePayloadFormat.DENSE_LOGITS_V0:
        raise ValueError(
            "dense target loader requires payload_format=dense_logits_v0, got "
            f"{manifest.payload_format.value}"
        )

    records_result = load_tome_records(artifact_dir / "records.jsonl")
    if records_result.blockers:
        raise ValueError(
            "teacher tome records are not loadable: "
            + ", ".join(records_result.blockers)
        )

    logits = _load_dense_logits(artifact_dir, manifest)
    _validate_loaded_logits(logits, manifest, record_count=len(records_result.records))
    return DenseTomeTargets(
        artifact_dir=artifact_dir,
        manifest=manifest,
        records=records_result.records,
        logits=logits,
    )


def _load_dense_logits(artifact_dir: Path, manifest: TomeManifest) -> np.ndarray:
    paths = _dense_payload_paths(manifest)
    arrays = [np.load(artifact_dir / path, allow_pickle=False) for path in paths]
    if len(arrays) == 1:
        return np.asarray(arrays[0])
    return np.concatenate(arrays, axis=0)


def _dense_payload_paths(manifest: TomeManifest) -> tuple[str, ...]:
    if manifest.shards:
        return tuple(shard.path for shard in manifest.shards)
    return (str(manifest.metadata.get("payload_path", DEFAULT_DENSE_LOGITS_PAYLOAD)),)


def _validate_loaded_logits(
    logits: np.ndarray,
    manifest: TomeManifest,
    *,
    record_count: int,
) -> None:
    blockers: list[str] = []
    if logits.ndim != 3:
        blockers.append(f"dense_logits_rank_invalid: rank={logits.ndim}")
    if not np.issubdtype(logits.dtype, np.floating):
        blockers.append(f"dense_logits_dtype_not_float: {logits.dtype}")
    if not np.all(np.isfinite(logits)):
        blockers.append("dense_logits_nonfinite")
    if logits.ndim == 3:
        if logits.shape[0] != record_count:
            blockers.append(
                f"dense_logits_record_dim_mismatch: logits={logits.shape[0]} "
                f"records={record_count}"
            )
        if (
            manifest.sequence_length is not None
            and logits.shape[1] != manifest.sequence_length
        ):
            blockers.append(
                f"dense_logits_sequence_dim_mismatch: logits={logits.shape[1]} "
                f"manifest={manifest.sequence_length}"
            )
        if (
            manifest.vocab_contract is not None
            and logits.shape[2] != manifest.vocab_contract.vocab_size
        ):
            blockers.append(
                f"dense_logits_vocab_dim_mismatch: logits={logits.shape[2]} "
                f"vocab={manifest.vocab_contract.vocab_size}"
            )
    if blockers:
        raise ValueError("teacher tome logits are not loadable: " + ", ".join(blockers))


def target_batch_from_dense_tome(targets: DenseTomeTargets) -> dict[str, Any]:
    return {
        "logits": targets.logits,
        "probabilities": targets.probabilities(),
        "example_ids": tuple(record.example_id for record in targets.records),
        "payload_format": targets.manifest.payload_format.value,
    }
