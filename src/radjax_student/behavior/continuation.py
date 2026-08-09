"""Durable, architecture-neutral behavioral continuation (P6.4).

The continuation envelope owns *source progress* and authority identities;
the model state remains the existing caller-bound learning-checkpoint v3.
Keeping these concerns separate makes a resume reject a stale schedule or
authority without inventing a second model serialization format.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.checkpoints.v3 import (
    JaxLearningCheckpointV3,
    save_learning_checkpoint_v3,
)
from radjax_student.contracts.hf import canonical_hf_json
from radjax_student.optimizers.protocols import JaxOptimizerBackend

BEHAVIOR_RUN_STATE_V1 = "behavior_run_state.v1"
BEHAVIOR_CONTINUATION_CHECKPOINT_V1 = "behavior_continuation_checkpoint.v1"
_FILES = ("run_state.json", "manifest.json")
_SHA256_PREFIX = "sha256:"
_SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


class BehaviorContinuationError(ValueError):
    """A malformed, stale, or non-atomic continuation envelope."""


class _CompletedPrefix(Sequence[str]):
    """O(1) view of the completed prefix of a canonical schedule.

    A run state only needs the prefix length to prove exactly-once progress;
    retaining a second tuple of all identities at every checkpoint caused
    quadratic copying over a long schedule.  The public serialization still
    expands this view to the historical identity list.
    """

    __slots__ = ("_schedule", "_length")

    def __init__(self, schedule: tuple[str, ...], length: int) -> None:
        self._schedule = schedule
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(itertools.islice(self._schedule, self._length))[index]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self._schedule[index]

    def __iter__(self):
        return itertools.islice(self._schedule, self._length)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompletedPrefix):
            return tuple(self) == tuple(other)
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))


def _digest(value: Any) -> str:
    return _SHA256_PREFIX + hashlib.sha256(canonical_hf_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return _SHA256_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BehaviorContinuationError(f"{name} must be an object")
    return {str(k): v for k, v in value.items()}


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BehaviorContinuationError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class BehaviorRunStateV1:
    """Canonical source schedule/progress state for an interrupted run.

    ``authority`` deliberately remains a neutral key/value block: callers can
    bind Contract, source, split, materialization, architecture, optimizer,
    HF, and carry identities without this module importing an architecture.
    ``completed_source_unit_identities`` is a prefix of the canonical schedule,
    which proves no source unit was silently skipped or replayed on resume.
    """

    run_id: str
    pass_id: str
    epoch: int
    next_item_index: int
    total_items: int
    source_batch_size: int
    checkpoint_interval_steps: int
    optimizer_step: int
    global_step: int
    retry_count: int
    authority: Mapping[str, str]
    scheduled_source_unit_identities: tuple[str, ...]
    completed_source_unit_identities: tuple[str, ...] = ()
    carry_policy_id: str = "reset_each_independent_example.v1"
    final_partial_batch_policy_id: str = "deterministic_singleton.v1"
    schema_version: str = BEHAVIOR_RUN_STATE_V1

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_RUN_STATE_V1:
            raise BehaviorContinuationError("unsupported behavioral run-state schema")
        _nonempty(self.run_id, "run_id")
        _nonempty(self.pass_id, "pass_id")
        if self.epoch < 0 or self.next_item_index < 0 or self.total_items < 0:
            raise BehaviorContinuationError("run coordinates must be nonnegative")
        if self.source_batch_size != 1:
            raise BehaviorContinuationError("P6.4 source batch size must be one")
        if self.checkpoint_interval_steps <= 0:
            raise BehaviorContinuationError("checkpoint interval must be positive")
        if min(self.optimizer_step, self.global_step, self.retry_count) < 0:
            raise BehaviorContinuationError(
                "step and retry counters must be nonnegative"
            )
        authority = _mapping(self.authority, "authority")
        if not authority or any(
            not k or not _nonempty(v, f"authority[{k}]") for k, v in authority.items()
        ):
            raise BehaviorContinuationError("authority identities must be nonempty")
        scheduled = tuple(self.scheduled_source_unit_identities)
        if isinstance(self.completed_source_unit_identities, _CompletedPrefix):
            if self.completed_source_unit_identities._schedule != scheduled:
                raise BehaviorContinuationError(
                    "completed ledger schedule does not match canonical schedule"
                )
            completed: Sequence[str] = self.completed_source_unit_identities
        else:
            completed = tuple(self.completed_source_unit_identities)
        if len(scheduled) != self.total_items:
            raise BehaviorContinuationError(
                "total_items does not match canonical schedule"
            )
        if (
            self.next_item_index != len(completed)
            or self.next_item_index > self.total_items
        ):
            raise BehaviorContinuationError(
                "next item index does not match completed ledger"
            )
        if any(
            not isinstance(v, str) or not v
            for v in itertools.chain(scheduled, completed)
        ):
            raise BehaviorContinuationError(
                "source identities must be nonempty strings"
            )
        if (
            not isinstance(completed, _CompletedPrefix)
            and scheduled[: len(completed)] != completed
        ):
            raise BehaviorContinuationError("completed ledger is not a schedule prefix")
        # The same source unit legitimately occurs once per explicit epoch;
        # exactly-once applies to each scheduled occurrence, not to the
        # underlying stable source identity across epochs.
        object.__setattr__(self, "authority", dict(sorted(authority.items())))
        object.__setattr__(self, "scheduled_source_unit_identities", scheduled)
        object.__setattr__(self, "completed_source_unit_identities", completed)

    @property
    def identity(self) -> str:
        return _digest(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload.pop("completed_source_unit_identities", None)
        payload.pop("optimizer_step", None)
        payload.pop("global_step", None)
        payload.pop("retry_count", None)
        payload.pop("next_item_index", None)
        payload.pop("epoch", None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["identity"] = self.identity
        return payload

    def _base_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "pass_id": self.pass_id,
            "epoch": self.epoch,
            "next_item_index": self.next_item_index,
            "total_items": self.total_items,
            "source_batch_size": self.source_batch_size,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "optimizer_step": self.optimizer_step,
            "global_step": self.global_step,
            "retry_count": self.retry_count,
            "authority": dict(self.authority),
            "scheduled_source_unit_identities": list(
                self.scheduled_source_unit_identities
            ),
            "completed_source_unit_identities": list(
                self.completed_source_unit_identities
            ),
            "carry_policy_id": self.carry_policy_id,
            "final_partial_batch_policy_id": self.final_partial_batch_policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorRunStateV1:
        p = _mapping(payload, "run state")
        identity = p.pop("identity", None)
        required = {
            "schema_version",
            "run_id",
            "pass_id",
            "epoch",
            "next_item_index",
            "total_items",
            "source_batch_size",
            "checkpoint_interval_steps",
            "optimizer_step",
            "global_step",
            "retry_count",
            "authority",
            "scheduled_source_unit_identities",
            "completed_source_unit_identities",
            "carry_policy_id",
            "final_partial_batch_policy_id",
        }
        if set(p) != required:
            raise BehaviorContinuationError(
                "run state fields are incomplete or unknown"
            )
        try:
            result = cls(
                **{
                    key: p[key]
                    for key in (
                        "run_id",
                        "pass_id",
                        "epoch",
                        "next_item_index",
                        "total_items",
                        "source_batch_size",
                        "checkpoint_interval_steps",
                        "optimizer_step",
                        "global_step",
                        "retry_count",
                        "authority",
                        "scheduled_source_unit_identities",
                        "completed_source_unit_identities",
                        "carry_policy_id",
                        "final_partial_batch_policy_id",
                        "schema_version",
                    )
                }
            )
        except (KeyError, TypeError) as exc:
            raise BehaviorContinuationError("run state fields are incomplete") from exc
        if identity != result.identity:
            raise BehaviorContinuationError("run state identity mismatch")
        return result

    def advance(
        self,
        source_unit_identity: str,
        *,
        epoch: int | None = None,
        optimizer_step: int | None = None,
        global_step: int | None = None,
        retry_count: int | None = None,
    ) -> BehaviorRunStateV1:
        if (
            self.next_item_index >= self.total_items
            or source_unit_identity
            != self.scheduled_source_unit_identities[self.next_item_index]
        ):
            raise BehaviorContinuationError(
                "source advancement is not the next canonical item"
            )
        from dataclasses import replace

        return replace(
            self,
            completed_source_unit_identities=_CompletedPrefix(
                self.scheduled_source_unit_identities, self.next_item_index + 1
            ),
            next_item_index=self.next_item_index + 1,
            epoch=self.epoch if epoch is None else epoch,
            optimizer_step=self.optimizer_step
            if optimizer_step is None
            else optimizer_step,
            global_step=self.global_step if global_step is None else global_step,
            retry_count=self.retry_count if retry_count is None else retry_count,
        )


@dataclass(frozen=True)
class BehaviorContinuationCheckpointV1:
    """Run state plus the identity of one canonical checkpoint-v3 payload."""

    run_state: BehaviorRunStateV1
    model_checkpoint_identity: str
    model_checkpoint_schema: str = "learning_checkpoint.v3"
    schema_version: str = BEHAVIOR_CONTINUATION_CHECKPOINT_V1

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_CONTINUATION_CHECKPOINT_V1:
            raise BehaviorContinuationError(
                "unsupported continuation checkpoint schema"
            )
        if not isinstance(self.run_state, BehaviorRunStateV1):
            raise TypeError("run_state must be BehaviorRunStateV1")
        if self.model_checkpoint_schema != "learning_checkpoint.v3":
            raise BehaviorContinuationError("unsupported model checkpoint schema")
        if not _SHA256_IDENTITY.fullmatch(self.model_checkpoint_identity):
            raise BehaviorContinuationError("model checkpoint identity must be sha256")

    @property
    def identity(self) -> str:
        return _digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_state": self.run_state.to_dict(),
            "model_checkpoint_schema": self.model_checkpoint_schema,
            "model_checkpoint_identity": self.model_checkpoint_identity,
        }
        if include_identity:
            result["identity"] = self.identity
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BehaviorContinuationCheckpointV1:
        p = _mapping(payload, "continuation checkpoint")
        identity = p.pop("identity", None)
        if set(p) != {
            "schema_version",
            "run_state",
            "model_checkpoint_schema",
            "model_checkpoint_identity",
        }:
            raise BehaviorContinuationError(
                "continuation checkpoint fields are incomplete or unknown"
            )
        try:
            result = cls(
                run_state=BehaviorRunStateV1.from_dict(p["run_state"]),
                model_checkpoint_identity=p["model_checkpoint_identity"],
                model_checkpoint_schema=p["model_checkpoint_schema"],
                schema_version=p["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise BehaviorContinuationError(
                "continuation checkpoint fields are incomplete"
            ) from exc
        if identity != result.identity:
            raise BehaviorContinuationError("continuation checkpoint identity mismatch")
        return result


@dataclass(frozen=True)
class BehaviorContinuationResultV1:
    """Deterministic result of one source-schedule continuation segment."""

    run_state: BehaviorRunStateV1
    status: str
    consumed_source_unit_identities: tuple[str, ...]
    checkpoint_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"complete", "stopped_at_boundary"}:
            raise BehaviorContinuationError("continuation result status is invalid")
        consumed = tuple(self.consumed_source_unit_identities)
        if consumed != self.run_state.completed_source_unit_identities:
            raise BehaviorContinuationError(
                "continuation result does not match the source ledger"
            )
        if any(
            not isinstance(value, str) or not value
            for value in self.checkpoint_identities
        ):
            raise BehaviorContinuationError(
                "continuation checkpoint identity is invalid"
            )
        object.__setattr__(self, "consumed_source_unit_identities", consumed)

    @property
    def state(self) -> BehaviorRunStateV1:
        """Compatibility spelling for orchestration callers."""
        return self.run_state

    @property
    def stopped_after_nonterminal_item(self) -> bool:
        return self.status == "stopped_at_boundary"


# Historical spelling used by the initial P6.4 orchestration draft.
BehaviorContinuationRunResultV1 = BehaviorContinuationResultV1


def run_behavior_continuation_v1(
    state: BehaviorRunStateV1,
    *,
    step: Callable[[str], Any] | None = None,
    checkpoint: Callable[[BehaviorRunStateV1, Any], str] | None = None,
    stop_after_steps: int | None = None,
    source_unit_identities: Sequence[str] | None = None,
    execute_source_unit: Callable[[str, BehaviorRunStateV1], Any] | None = None,
    write_checkpoint: Callable[[BehaviorRunStateV1], str] | None = None,
    stop_after_items: int | None = None,
) -> BehaviorContinuationResultV1:
    """Advance a canonical B=1 source schedule with exactly-once semantics.

    ``step`` is called only for the next schedule identity and must return
    successfully before that identity is recorded.  A stop request is honored
    after a completed batch, while checkpoint callbacks are made only at the
    configured cadence or terminal boundary.  Consequently a caller can resume
    from the last durable envelope without replaying a committed source unit.
    The callback owns model-v3 persistence; this function remains architecture
    and optimizer neutral.
    """

    if not isinstance(state, BehaviorRunStateV1):
        raise TypeError("state must be BehaviorRunStateV1")
    if source_unit_identities is not None:
        if tuple(source_unit_identities) != state.scheduled_source_unit_identities:
            raise BehaviorContinuationError("source schedule does not match run state")
    execute_with_state = execute_source_unit
    if step is None and execute_with_state is None:
        raise TypeError("a source-unit step callback is required")
    if checkpoint is None and write_checkpoint is not None:

        def checkpoint_from_state(current: BehaviorRunStateV1, _outcome: Any) -> str:
            return write_checkpoint(current)

        checkpoint = checkpoint_from_state
    if stop_after_items is not None:
        stop_after_steps = stop_after_items
    if stop_after_steps is not None and (
        not isinstance(stop_after_steps, int)
        or isinstance(stop_after_steps, bool)
        or stop_after_steps < 1
    ):
        raise BehaviorContinuationError("stop_after_steps must be a positive integer")
    current = state
    # Keep the source ledger in an append-only list while a segment is being
    # executed.  Materialising ``tuple(consumed)`` for every B=1 callback made
    # a long (64 epoch) schedule quadratic in both time and allocations.  A
    # durable run state is still materialised at checkpoint/stop/terminal
    # boundaries, preserving the public tuple representation and its exact
    # prefix validation.
    consumed: list[str] = list(state.completed_source_unit_identities)
    checkpoints: list[str] = []
    segment_steps = 0
    next_item_index = state.next_item_index
    while next_item_index < state.total_items:
        source_identity = state.scheduled_source_unit_identities[next_item_index]
        outcome = (
            execute_with_state(source_identity, current)
            if execute_with_state is not None
            else step(source_identity)  # type: ignore[misc]
        )
        consumed.append(source_identity)
        next_item_index += 1
        segment_steps += 1
        metrics = outcome if isinstance(outcome, Mapping) else {}
        base_count = len(state.completed_source_unit_identities)
        default_step = state.optimizer_step + len(consumed) - base_count
        default_global = state.global_step + len(consumed) - base_count
        terminal = next_item_index == state.total_items
        due = default_step % state.checkpoint_interval_steps == 0
        materialize = (
            due
            or terminal
            or (stop_after_steps is not None and segment_steps >= stop_after_steps)
        )
        if materialize:
            current = BehaviorRunStateV1(
                run_id=state.run_id,
                pass_id=state.pass_id,
                epoch=int(metrics.get("epoch", current.epoch)),
                next_item_index=next_item_index,
                total_items=state.total_items,
                source_batch_size=state.source_batch_size,
                checkpoint_interval_steps=state.checkpoint_interval_steps,
                optimizer_step=int(metrics.get("optimizer_step", default_step)),
                global_step=int(metrics.get("global_step", default_global)),
                retry_count=int(metrics.get("retry_count", state.retry_count)),
                authority=state.authority,
                scheduled_source_unit_identities=state.scheduled_source_unit_identities,
                completed_source_unit_identities=_CompletedPrefix(
                    state.scheduled_source_unit_identities, next_item_index
                ),
                carry_policy_id=state.carry_policy_id,
                final_partial_batch_policy_id=state.final_partial_batch_policy_id,
            )
        if checkpoint is not None and (due or terminal):
            identity = checkpoint(current, outcome)
            if not isinstance(identity, str) or not identity.startswith(_SHA256_PREFIX):
                raise BehaviorContinuationError(
                    "checkpoint callback must return a sha256 identity"
                )
            checkpoints.append(identity)
        if (
            stop_after_steps is not None
            and segment_steps >= stop_after_steps
            and not terminal
        ):
            return BehaviorContinuationResultV1(
                current, "stopped_at_boundary", tuple(consumed), tuple(checkpoints)
            )
    if current.next_item_index != next_item_index:
        current = BehaviorRunStateV1(
            run_id=state.run_id,
            pass_id=state.pass_id,
            epoch=state.epoch,
            next_item_index=next_item_index,
            total_items=state.total_items,
            source_batch_size=state.source_batch_size,
            checkpoint_interval_steps=state.checkpoint_interval_steps,
            optimizer_step=state.optimizer_step
            + len(consumed)
            - len(state.completed_source_unit_identities),
            global_step=state.global_step
            + len(consumed)
            - len(state.completed_source_unit_identities),
            retry_count=state.retry_count,
            authority=state.authority,
            scheduled_source_unit_identities=state.scheduled_source_unit_identities,
            completed_source_unit_identities=_CompletedPrefix(
                state.scheduled_source_unit_identities, next_item_index
            ),
            carry_policy_id=state.carry_policy_id,
            final_partial_batch_policy_id=state.final_partial_batch_policy_id,
        )
    return BehaviorContinuationResultV1(
        current, "complete", tuple(consumed), tuple(checkpoints)
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_behavior_continuation_checkpoint_v1(
    checkpoint: BehaviorContinuationCheckpointV1,
    directory: Path,
    *,
    model_checkpoint: JaxLearningCheckpointV3,
    optimizer: JaxOptimizerBackend,
) -> BehaviorContinuationCheckpointV1:
    """Atomically write v3 model state and its source-progress envelope.

    Existing nonempty destinations are never mutated. A crash can leave only a
    hidden sibling temporary directory; no reader treats that directory as a
    committed checkpoint.
    """

    if not isinstance(checkpoint, BehaviorContinuationCheckpointV1):
        raise TypeError("checkpoint must be BehaviorContinuationCheckpointV1")
    if os.path.lexists(directory) and (
        not directory.is_dir() or any(directory.iterdir())
    ):
        raise BehaviorContinuationError("continuation destination already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent)
    )
    try:
        model_dir = temporary / "model"
        saved_model = save_learning_checkpoint_v3(
            model_checkpoint, model_dir, optimizer=optimizer
        )
        raw_model_identity = saved_model.integrity.get("manifest_digest")
        observed_model_identity = (
            None if raw_model_identity is None else _SHA256_PREFIX + raw_model_identity
        )
        if observed_model_identity != checkpoint.model_checkpoint_identity:
            raise BehaviorContinuationError(
                "model checkpoint identity does not match envelope"
            )
        (temporary / "run_state.json").write_bytes(
            canonical_hf_json(checkpoint.run_state.to_dict())
        )
        manifest = {
            "schema_version": BEHAVIOR_CONTINUATION_CHECKPOINT_V1,
            "files": {
                name: _file_digest(temporary / name) for name in ("run_state.json",)
            },
            "model": {
                "schema_version": saved_model.schema_version,
                "identity": observed_model_identity,
                "files": {
                    name: _file_digest(model_dir / name)
                    for name in sorted(
                        p.name for p in model_dir.iterdir() if p.is_file()
                    )
                },
            },
            "run_state_identity": checkpoint.run_state.identity,
            "identity": checkpoint.identity,
        }
        (temporary / "manifest.json").write_bytes(canonical_hf_json(manifest))
        for path in (temporary / "run_state.json", temporary / "manifest.json"):
            _fsync_file(path)
        _fsync_directory(temporary)
        if directory.exists():
            directory.rmdir()
        os.replace(temporary, directory)
        _fsync_directory(directory.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint


def load_behavior_continuation_checkpoint_v1(
    directory: Path,
    *,
    restore_model: Callable[[Path], JaxLearningCheckpointV3],
    expected_run_identity: str | None = None,
) -> tuple[BehaviorContinuationCheckpointV1, JaxLearningCheckpointV3]:
    """Read and verify every envelope/model byte before allowing continuation."""

    if not directory.is_dir():
        raise BehaviorContinuationError("continuation checkpoint directory is absent")
    top_level = {p.name for p in directory.iterdir()}
    if top_level != {"run_state.json", "manifest.json", "model"}:
        raise BehaviorContinuationError(
            "continuation checkpoint has missing or extra top-level files"
        )
    try:
        payload = json.loads((directory / "manifest.json").read_text())
        state = BehaviorRunStateV1.from_dict(
            json.loads((directory / "run_state.json").read_text())
        )
        checkpoint = BehaviorContinuationCheckpointV1(
            run_state=state,
            model_checkpoint_identity=payload["model"]["identity"],
            model_checkpoint_schema=payload["model"]["schema_version"],
            schema_version=payload["schema_version"],
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BehaviorContinuationError(
            "continuation checkpoint metadata is invalid"
        ) from exc
    if expected_run_identity is not None and state.identity != expected_run_identity:
        raise BehaviorContinuationError("continuation run authority identity mismatch")
    if (
        payload.get("identity") != checkpoint.identity
        or payload.get("run_state_identity") != state.identity
    ):
        raise BehaviorContinuationError("continuation manifest identity mismatch")
    if set(payload) != {
        "schema_version",
        "files",
        "model",
        "run_state_identity",
        "identity",
    }:
        raise BehaviorContinuationError("continuation manifest has unknown fields")
    if payload.get("files", {}).get("run_state.json") != _file_digest(
        directory / "run_state.json"
    ):
        raise BehaviorContinuationError("continuation run-state bytes are tampered")
    model_dir = directory / "model"
    if not model_dir.is_dir():
        raise BehaviorContinuationError("continuation model checkpoint is absent")
    expected_files = payload.get("model", {}).get("files")
    if not isinstance(expected_files, Mapping):
        raise BehaviorContinuationError("continuation model inventory is absent")
    observed_files = {
        p.name: _file_digest(p) for p in sorted(model_dir.iterdir()) if p.is_file()
    }
    if observed_files != dict(expected_files):
        raise BehaviorContinuationError(
            "continuation model bytes are missing or tampered"
        )
    try:
        restored = restore_model(model_dir)
    except Exception as exc:
        raise BehaviorContinuationError(
            "model checkpoint v3 failed caller-bound restore"
        ) from exc
    try:
        model_manifest = json.loads((model_dir / "manifest.json").read_text())
        raw_observed = model_manifest["integrity"]["manifest_digest"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BehaviorContinuationError(
            "model checkpoint v3 manifest integrity is invalid"
        ) from exc
    observed = None if raw_observed is None else _SHA256_PREFIX + raw_observed
    if observed != checkpoint.model_checkpoint_identity:
        raise BehaviorContinuationError("restored model identity differs from envelope")
    return checkpoint, restored


def restore_behavior_continuation_with_lifecycle(
    directory: Path, lifecycle: Any
) -> tuple[BehaviorContinuationCheckpointV1, Any]:
    """Convenience adapter for :class:`JaxLearningLifecycle` caller binding."""

    checkpoint, restored = load_behavior_continuation_checkpoint_v1(
        directory,
        restore_model=lambda path: lifecycle.restore_from_checkpoint(path).checkpoint(),
        expected_run_identity=None,
    )
    return checkpoint, lifecycle.with_checkpoint(restored)
