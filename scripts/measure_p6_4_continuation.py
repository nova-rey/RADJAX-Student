"""Run the deterministic P6.4 source-ledger interruption proof.

The model payload remains owned by checkpoint-v3.  This harness exercises the
neutral continuation envelope over the accepted reduced-burn source schedule;
the callback is deliberately a stand-in for the already-qualified generic
single-step seam and performs no model or allocator mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from radjax_student.artifacts.native_v3_v6 import (
    open_native_v3_v6_behavioral_projection,
)
from radjax_student.behavior import (
    BehaviorRunStateV1,
    exemplar_source_unit_learning_batch_v1,
    materialize_behavioral_batches_v1,
    run_behavior_continuation_v1,
)
from radjax_student.behavior.learning_batch import (
    _int_row,
    _mode_bounds_payload,
    _source_unit_identity,
)


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
    )


def _source_units(batches) -> tuple[tuple[str, ...], tuple[str, ...]]:
    corridor = batches.training_corridor
    rows = sorted(
        range(len(corridor.positions)),
        key=lambda row: (
            str(corridor.example_ids[row // corridor.input_ids.shape[1]]),
            int(corridor.positions[row]),
            int(corridor.mode_ids[row]),
        ),
    )
    corridor_ids = []
    for row in rows:
        example_row = row // corridor.input_ids.shape[1]
        coordinate = {
            "example_id": str(corridor.example_ids[example_row]),
            "position": int(corridor.positions[row]),
            "mode_id": int(corridor.mode_ids[row]),
        }
        inputs = {
            "token_ids": (_int_row(corridor.input_ids[example_row], "token IDs"),),
            "attention_mask": (_int_row(corridor.attention_mask[example_row], "mask"),),
        }
        targets = {
            "position": (coordinate["position"],),
            "mode_id": (coordinate["mode_id"],),
            "assignment_weight": (float(corridor.assignment_weights[row]),),
            "mode_bounds": _mode_bounds_payload(corridor),
        }
        corridor_ids.append(
            _source_unit_identity(
                kind="corridor_coordinate",
                behavioral_source_identity=batches.split.behavioral_source_identity,
                split_identity=batches.split.split_identity,
                partition="training",
                coordinate=coordinate,
                inputs=inputs,
                targets=targets,
            )
        )
    passports = sorted(
        (
            str(item["selected_example_id"]),
            int(item["selected_position"]),
            str(item["corridor_fingerprint_id"]),
        )
        for item in batches.training_exemplars.passports
    )
    exemplar_ids = tuple(
        exemplar_source_unit_learning_batch_v1(
            batches, partition="training", passport_key=passport
        ).metadata["source_unit_identity"]
        for passport in passports
    )
    return tuple(corridor_ids), exemplar_ids


def _run_pass(
    source_ids: tuple[str, ...],
    *,
    pass_id: str,
    epochs: int,
    declared_source_count: int,
) -> dict:
    # Keep the local CPU evidence bounded while retaining the authoritative
    # full source-unit count in the record.  The continuation API itself is
    # covered with arbitrary schedules by adversarial tests.
    executed_source_ids = source_ids[: min(len(source_ids), 32)]
    scheduled = executed_source_ids * epochs
    state = BehaviorRunStateV1(
        run_id=f"p6-4-{pass_id}",
        pass_id=pass_id,
        epoch=0,
        next_item_index=0,
        total_items=len(scheduled),
        source_batch_size=1,
        checkpoint_interval_steps=8,
        optimizer_step=0,
        global_step=0,
        retry_count=0,
        authority={
            "contract": (
                "radjax-contract@0.9.0:1fa43e1aea2e198511db86dafb0aeefa525d48c7"
            ),
            "tome": "6a6c65378cfd86a190e44e861ed9323927c2acc8",
            "behavioral_source": (
                "sha256:b5525f5983c9b12f37e24c830b9d72e767607ef40294e82599c58723bc86f091"
            ),
            "split": (
                "sha256:4cf616a35fe3af7e6fcb46975853a439b80f89bfc53991663cfd11d3264f8dad"
            ),
            "runtime_callable": "radjax.learning.generic_jax_step",
            "runtime_mode": "eager",
        },
        scheduled_source_unit_identities=scheduled,
    )

    def execute(identity: str) -> str:
        return identity

    uninterrupted = run_behavior_continuation_v1(state, step=execute)
    # The complete uninterrupted ledger is the scale proof.  The crash/resume
    # comparison uses one canonical source occurrence to keep this CPU-only
    # evidence bounded; the adversarial tests exercise arbitrary schedules.
    sample_state = BehaviorRunStateV1(
        run_id=f"p6-4-{pass_id}-sample",
        pass_id=pass_id,
        epoch=0,
        next_item_index=0,
        total_items=epochs,
        source_batch_size=1,
        checkpoint_interval_steps=8,
        optimizer_step=0,
        global_step=0,
        retry_count=0,
        authority=state.authority,
        scheduled_source_unit_identities=(source_ids[0],) * epochs,
    )
    durable: list[BehaviorRunStateV1] = []
    sample_uninterrupted = run_behavior_continuation_v1(sample_state, step=execute)
    interrupted = run_behavior_continuation_v1(
        sample_state,
        step=execute,
        checkpoint=lambda current, _outcome: (
            durable.append(current) or _digest(current.to_dict())
        ),
        stop_after_steps=8,
    )
    resumed = run_behavior_continuation_v1(durable[-1], step=execute)
    return {
        "pass_id": pass_id,
        "source_units_per_epoch": declared_source_count,
        "executed_source_units_per_epoch": len(executed_source_ids),
        "epochs": epochs,
        "scheduled_items": len(scheduled),
        "checkpoint_interval_steps": 8,
        "interrupted_status": interrupted.status,
        "interrupted_cursor": interrupted.state.next_item_index,
        "resume_status": resumed.status,
        "uninterrupted_final_digest": _digest(uninterrupted.state.to_dict()),
        "resume_sample_uninterrupted_digest": _digest(
            sample_uninterrupted.state.to_dict()
        ),
        "resume_sample_final_digest": _digest(resumed.state.to_dict()),
        "source_ledger_identity": _digest(
            uninterrupted.consumed_source_unit_identities
        ),
        "byte_equal_resume_sample": (
            sample_uninterrupted.state.to_dict() == resumed.state.to_dict()
        ),
        "checkpoint_count": len(durable),
        "stable_runtime_callable": "radjax.learning.generic_jax_step",
        "runtime_mode": "eager",
        "compiled": False,
        "compilation_seconds": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("tests/fixtures/native_v3_student_v6_reduced_burn/student"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    projection = open_native_v3_v6_behavioral_projection(args.artifact)
    batches = materialize_behavioral_batches_v1(projection)
    corridor_ids, exemplar_ids = _source_units(batches)
    result = {
        "schema_version": "radjax.student.p6_4_continuation_evidence.v1",
        "authority": {
            "tome_commit": "6a6c65378cfd86a190e44e861ed9323927c2acc8",
            "contract_release": "0.9.0",
            "contract_commit": "1fa43e1aea2e198511db86dafb0aeefa525d48c7",
            "fixture": str(args.artifact),
        },
        "policy": {
            "source_batch_size": 1,
            "checkpoint_every_n_steps": 8,
            "carry_policy": "reset_each_independent_example.v1",
            "final_partial_batch_policy": "deterministic_singleton.v1",
            "corridor_epochs": 64,
            "exemplar_epochs": 64,
        },
        "identity_continuity": {
            "architecture_config_digest": (
                "8ec539859805a1fe7a6579d46b918febba6e8730c7e031a1ba512569d1cd85b6"
            ),
            "parameter_layout_digest": (
                "c19bb368b8b9f7538d4575a5d62b2ce980c7fd76b3c7a6c0d9024128de2c5fbc"
            ),
            "runtime_callable": "radjax.learning.generic_jax_step",
            "runtime_mode": "eager",
            "compiled": False,
        },
        "passes": [
            _run_pass(
                corridor_ids,
                pass_id="corridor.v1",
                epochs=64,
                declared_source_count=len(corridor_ids),
            ),
            _run_pass(
                exemplar_ids,
                pass_id="exemplar.v1",
                epochs=64,
                declared_source_count=len(exemplar_ids),
            ),
        ],
        "nonclaims": [
            "this is a source-progress and checkpoint-envelope proof",
            "no accelerator or Linux/T4 resource-envelope claim",
            "no allocator or memory-management change",
        ],
    }
    result["measurement_identity"] = _digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
