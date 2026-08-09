"""Adversarial tests for the P6.4 source-progress continuation envelope."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from radjax_student.behavior.continuation import (
    BehaviorContinuationCheckpointV1,
    BehaviorContinuationError,
    BehaviorRunStateV1,
    load_behavior_continuation_checkpoint_v1,
    run_behavior_continuation_v1,
    save_behavior_continuation_checkpoint_v1,
)


def _state() -> BehaviorRunStateV1:
    return BehaviorRunStateV1(
        run_id="run-1",
        pass_id="corridor.v1",
        epoch=0,
        next_item_index=0,
        total_items=3,
        source_batch_size=1,
        checkpoint_interval_steps=8,
        optimizer_step=0,
        global_step=0,
        retry_count=0,
        authority={"contract": "sha256:contract", "source": "sha256:source"},
        scheduled_source_unit_identities=("sha256:a", "sha256:b", "sha256:c"),
    )


def _checkpoint() -> BehaviorContinuationCheckpointV1:
    return BehaviorContinuationCheckpointV1(
        run_state=_state().advance("sha256:a"),
        model_checkpoint_identity="sha256:" + "1" * 64,
    )


def test_run_state_requires_exact_prefix_and_rejects_skip_or_replay() -> None:
    state = _state()
    advanced = state.advance("sha256:a", optimizer_step=1, global_step=1)
    assert advanced.next_item_index == 1
    assert advanced.completed_source_unit_identities == ("sha256:a",)
    assert BehaviorRunStateV1.from_dict(advanced.to_dict()) == advanced
    with pytest.raises(BehaviorContinuationError, match="next canonical item"):
        state.advance("sha256:b")
    with pytest.raises(BehaviorContinuationError, match="next canonical item"):
        advanced.advance("sha256:a")

    payload = advanced.to_dict()
    payload["completed_source_unit_identities"] = ["sha256:b"]
    with pytest.raises(BehaviorContinuationError, match="schedule prefix"):
        BehaviorRunStateV1.from_dict(payload)


def test_runner_advances_exactly_once_and_stops_after_nonterminal_item() -> None:
    state = _state()
    calls: list[str] = []
    writes: list[tuple[int, tuple[str, ...]]] = []
    first = run_behavior_continuation_v1(
        state,
        step=lambda source: calls.append(source) or object(),
        checkpoint=lambda current, outcome: (
            writes.append(
                (current.next_item_index, current.completed_source_unit_identities)
            )
            or f"sha256:{current.next_item_index:064x}"
        ),
        stop_after_steps=1,
    )
    assert first.status == "stopped_at_boundary"
    assert first.consumed_source_unit_identities == ("sha256:a",)
    assert calls == ["sha256:a"]
    assert writes == []

    resumed = run_behavior_continuation_v1(
        first.state,
        step=lambda source: calls.append(source) or object(),
        checkpoint=lambda current, outcome: f"sha256:{current.next_item_index:064x}",
    )
    assert resumed.status == "complete"
    assert (
        resumed.consumed_source_unit_identities
        == state.scheduled_source_unit_identities
    )
    assert calls == ["sha256:a", "sha256:b", "sha256:c"]


def test_envelope_writer_is_atomic_and_reader_rejects_tampering(
    tmp_path, monkeypatch
) -> None:
    import radjax_student.behavior.continuation as continuation

    def fake_save(checkpoint, directory, *, optimizer):
        directory.mkdir(parents=True)
        (directory / "parameters.npz").write_bytes(b"model")
        (directory / "manifest.json").write_text(
            json.dumps({"integrity": {"manifest_digest": "1" * 64}})
        )
        return SimpleNamespace(
            schema_version="learning_checkpoint.v3",
            integrity={"manifest_digest": "1" * 64},
        )

    monkeypatch.setattr(continuation, "save_learning_checkpoint_v3", fake_save)
    destination = tmp_path / "continuation"
    envelope = _checkpoint()
    # The production writer stores the raw v3 digest as a sha256 identity.
    envelope = BehaviorContinuationCheckpointV1(
        run_state=envelope.run_state,
        model_checkpoint_identity="sha256:" + "1" * 64,
    )
    saved = save_behavior_continuation_checkpoint_v1(
        envelope, destination, model_checkpoint=object(), optimizer=object()
    )
    assert saved.identity == envelope.identity
    assert (destination / "model" / "parameters.npz").read_bytes() == b"model"
    assert not list(tmp_path.glob(".continuation.tmp-*"))

    def restore(path):
        assert (path / "parameters.npz").read_bytes() == b"model"
        return SimpleNamespace(integrity={"manifest_digest": "1" * 64})

    loaded, restored = load_behavior_continuation_checkpoint_v1(
        destination, restore_model=restore
    )
    assert loaded == envelope
    assert restored.integrity["manifest_digest"] == "1" * 64

    (destination / "run_state.json").write_text(
        json.dumps({**envelope.run_state.to_dict(), "next_item_index": 2})
    )
    with pytest.raises(BehaviorContinuationError, match="metadata|identity|tampered"):
        load_behavior_continuation_checkpoint_v1(destination, restore_model=restore)

    (destination / "run_state.json").write_bytes(
        json.dumps(envelope.run_state.to_dict()).encode()
    )
    (destination / "model" / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(BehaviorContinuationError, match="tampered"):
        load_behavior_continuation_checkpoint_v1(destination, restore_model=restore)


def test_writer_refuses_existing_destination_without_mutation(
    tmp_path, monkeypatch
) -> None:
    import radjax_student.behavior.continuation as continuation

    monkeypatch.setattr(
        continuation,
        "save_learning_checkpoint_v3",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write")),
    )
    destination = tmp_path / "continuation"
    destination.mkdir()
    marker = destination / "marker"
    marker.write_bytes(b"preserve")
    with pytest.raises(BehaviorContinuationError, match="already exists"):
        save_behavior_continuation_checkpoint_v1(
            _checkpoint(), destination, model_checkpoint=object(), optimizer=object()
        )
    assert marker.read_bytes() == b"preserve"


def test_callback_runner_stops_at_boundary_and_resumes_without_replay() -> None:
    state = BehaviorRunStateV1(
        run_id="run-2",
        pass_id="exemplar.v1",
        epoch=0,
        next_item_index=0,
        total_items=5,
        source_batch_size=1,
        checkpoint_interval_steps=2,
        optimizer_step=0,
        global_step=0,
        retry_count=0,
        authority={"contract": "sha256:contract", "source": "sha256:source"},
        scheduled_source_unit_identities=tuple(f"sha256:{c}" for c in "abcde"),
    )
    seen: list[str] = []
    durable: list[BehaviorRunStateV1] = []
    first = run_behavior_continuation_v1(
        state,
        step=lambda identity: seen.append(identity) or identity,
        checkpoint=lambda current, _: (
            durable.append(current) or f"sha256:{current.next_item_index:064d}"
        ),
        stop_after_steps=2,
    )
    assert first.status == "stopped_at_boundary"
    assert seen == [f"sha256:{c}" for c in "ab"]
    assert [s.next_item_index for s in durable] == [2]
    resumed = run_behavior_continuation_v1(
        durable[-1],
        step=lambda identity: seen.append(identity) or identity,
        checkpoint=lambda current, _: f"sha256:{current.next_item_index:064d}",
    )
    assert resumed.status == "complete"
    assert seen == [f"sha256:{c}" for c in "abcde"]
    assert len(set(seen)) == 5


def test_full_reduced_burn_schedule_uses_linear_ledger_and_resumes() -> None:
    """The accepted 64-epoch source schedule must not copy a growing ledger.

    The reduced-burn corridor has 32 source units (2048 occurrences over 64
    epochs); this test uses the complete corridor-plus-exemplar-sized schedule
    and exercises a durable checkpoint/resume boundary without relying on a
    model backend.
    """
    corridor = tuple(f"sha256:c{i:04d}" for i in range(32))
    exemplar = tuple(f"sha256:e{i:04d}" for i in range(33))
    schedule = (corridor * 64) + (exemplar * 64)
    state = BehaviorRunStateV1(
        run_id="full-reduced-burn",
        pass_id="corridor-exemplar.v1",
        epoch=0,
        next_item_index=0,
        total_items=len(schedule),
        source_batch_size=1,
        checkpoint_interval_steps=8,
        optimizer_step=0,
        global_step=0,
        retry_count=0,
        authority={"contract": "sha256:contract", "source": "sha256:source"},
        scheduled_source_unit_identities=schedule,
    )
    durable: list[BehaviorRunStateV1] = []

    uninterrupted = run_behavior_continuation_v1(
        state,
        step=lambda _identity: None,
    )
    interrupted = run_behavior_continuation_v1(
        state,
        step=lambda _identity: None,
        checkpoint=lambda current, _outcome: (
            durable.append(current) or f"sha256:{current.next_item_index:064x}"
        ),
        stop_after_steps=100,
    )
    assert interrupted.status == "stopped_at_boundary"
    assert interrupted.state.next_item_index == 100
    assert durable[-1].next_item_index == 96

    resumed = run_behavior_continuation_v1(
        durable[-1],
        step=lambda _identity: None,
    )
    assert resumed.status == "complete"
    assert resumed.state.next_item_index == len(schedule)
    assert resumed.state.completed_source_unit_identities == schedule
    # The final state from a checkpoint resume is exactly the uninterrupted
    # source ledger and counters, despite the durable stop occurring earlier.
    assert resumed.state.to_dict() == uninterrupted.state.to_dict()
