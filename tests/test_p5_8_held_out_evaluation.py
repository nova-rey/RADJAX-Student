"""P5.8 deterministic leakage-free held-out evaluation evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.behavior import (  # noqa: E402
    HeldOutEvaluationBindingV1,
    HeldOutEvaluationError,
    evaluate_held_out_behavior_v1,
    materialize_behavioral_batches_v1,
    replay_held_out_evaluation_v1,
    run_exemplar_pass_v1,
)
from tests.test_p5_4_behavior_materialization import _projection  # noqa: E402
from tests.test_p5_7_exemplar_pass import _setup  # noqa: E402

pytestmark = pytest.mark.jax


def _evaluation_values():
    training = _setup()
    batches = materialize_behavioral_batches_v1(_projection())
    final = run_exemplar_pass_v1(**training).checkpoint
    return dict(
        corridor_checkpoint=training["predecessor"],
        final_checkpoint=final,
        binding=HeldOutEvaluationBindingV1.from_final_checkpoint(final, batches),
        expected_batches=batches,
        training_corridor=batches.training_corridor,
        training_exemplars=batches.training_exemplars,
        held_out_corridor=batches.held_out_corridor,
        held_out_exemplars=batches.held_out_exemplars,
        forward=training["forward"],
    )


def test_p5_8_evaluates_each_held_out_value_once_without_model_or_optimizer_updates():
    values = _evaluation_values()
    before_parameters = jax.tree_util.tree_map(
        lambda value: jnp.array(value), values["final_checkpoint"].parameters
    )
    before_optimizer = values["final_checkpoint"].optimizer_state
    report = evaluate_held_out_behavior_v1(**values)
    assert len(report.corridor_coordinates) == len(
        values["held_out_corridor"].positions
    )
    assert len(set(report.corridor_coordinates)) == len(report.corridor_coordinates)
    assert len(report.exemplar_passport_keys) == len(
        values["held_out_exemplars"].passports
    )
    assert len(set(report.exemplar_passport_keys)) == len(report.exemplar_passport_keys)
    assert all(jnp.isfinite(value) for _, value in report.corridor_metrics)
    assert all(jnp.isfinite(value) for _, value in report.exemplar_metrics)
    assert values["final_checkpoint"].optimizer_state == before_optimizer
    for before, after in zip(
        jax.tree_util.tree_leaves(before_parameters),
        jax.tree_util.tree_leaves(values["final_checkpoint"].parameters),
        strict=True,
    ):
        assert bool(jnp.array_equal(before, after))


def test_p5_8_report_replay_is_exact_and_binds_final_checkpoint_identity():
    values = _evaluation_values()
    report = evaluate_held_out_behavior_v1(**values)
    replay = replay_held_out_evaluation_v1(expected=report, **values)
    assert replay.identity == report.identity
    with pytest.raises(
        HeldOutEvaluationError, match="final checkpoint identity mismatch"
    ):
        evaluate_held_out_behavior_v1(
            **{
                **values,
                "binding": replace(
                    values["binding"], final_checkpoint_identity="sha256:" + "0" * 64
                ),
            }
        )


def test_p5_8_rejects_leakage_in_training_batches_or_incomplete_training_cursors():
    values = _evaluation_values()
    leaked = replace(
        values["training_exemplars"],
        example_ids=values["held_out_exemplars"].example_ids,
    )
    with pytest.raises(HeldOutEvaluationError, match="leaks into training"):
        evaluate_held_out_behavior_v1(**{**values, "training_exemplars": leaked})
    incomplete = replace(values["corridor_checkpoint"], cursor=0)
    final = replace(
        values["final_checkpoint"],
        binding=replace(
            values["final_checkpoint"].binding,
            predecessor_checkpoint_identity=incomplete.identity,
        ),
    )
    with pytest.raises(HeldOutEvaluationError, match="corridor training cursor"):
        evaluate_held_out_behavior_v1(
            **{
                **values,
                "corridor_checkpoint": incomplete,
                "final_checkpoint": final,
                "binding": HeldOutEvaluationBindingV1.from_final_checkpoint(
                    final, values["expected_batches"]
                ),
            }
        )


def test_p5_8_rejects_duplicate_held_out_evidence_and_changed_continuity():
    values = _evaluation_values()
    batch = values["held_out_exemplars"]
    duplicate = replace(
        batch,
        passports=(batch.passports[0], batch.passports[0]),
        sparse_targets=(batch.sparse_targets[0], batch.sparse_targets[0]),
    )
    with pytest.raises(
        HeldOutEvaluationError, match="held-out exemplars are not unique"
    ):
        evaluate_held_out_behavior_v1(**{**values, "held_out_exemplars": duplicate})
    with pytest.raises(HeldOutEvaluationError, match="continuity authority mismatch"):
        evaluate_held_out_behavior_v1(
            **{
                **values,
                "binding": replace(
                    values["binding"], split_identity="sha256:" + "9" * 64
                ),
            }
        )


@pytest.mark.parametrize("surface", ("held_out_corridor", "held_out_exemplars"))
def test_p5_8_rejects_partial_held_out_evidence_against_bound_expected_sets(surface):
    values = _evaluation_values()
    batch = values[surface]
    if surface == "held_out_corridor":
        partial = replace(
            batch,
            input_ids=batch.input_ids[:1],
            attention_mask=batch.attention_mask[:1],
            example_ids=batch.example_ids[:1],
            example_indices=batch.example_indices[:1],
            positions=batch.positions[:1],
            mode_ids=batch.mode_ids[:1],
            assignment_weights=batch.assignment_weights[:1],
        )
        message = "held-out corridor evidence is incomplete or substituted"
    else:
        partial = replace(
            batch,
            input_ids=batch.input_ids[:1],
            attention_mask=batch.attention_mask[:1],
            example_ids=batch.example_ids[:1],
            selected_example_indices=batch.selected_example_indices[:1],
            selected_positions=batch.selected_positions[:1],
            sparse_targets=batch.sparse_targets[:1],
            passports=batch.passports[:1],
        )
        message = "held-out exemplar evidence is incomplete or substituted"
    with pytest.raises(HeldOutEvaluationError, match=message):
        evaluate_held_out_behavior_v1(**{**values, surface: partial})


def test_p5_8_rejects_substituted_held_out_passport_against_bound_expected_set():
    values = _evaluation_values()
    batch = values["held_out_exemplars"]
    substituted = replace(
        batch,
        passports=(
            {**batch.passports[0], "corridor_fingerprint_id": "substituted"},
            batch.passports[1],
        ),
    )
    with pytest.raises(
        HeldOutEvaluationError,
        match="held-out exemplar evidence is incomplete or substituted",
    ):
        evaluate_held_out_behavior_v1(**{**values, "held_out_exemplars": substituted})


def test_p5_8_rejects_partial_expected_materialization_before_binding():
    values = _evaluation_values()
    expected = values["expected_batches"]
    partial = replace(
        expected.held_out_corridor,
        input_ids=expected.held_out_corridor.input_ids[:1],
        attention_mask=expected.held_out_corridor.attention_mask[:1],
        example_ids=expected.held_out_corridor.example_ids[:1],
        example_indices=expected.held_out_corridor.example_indices[:1],
        positions=expected.held_out_corridor.positions[:1],
        mode_ids=expected.held_out_corridor.mode_ids[:1],
        assignment_weights=expected.held_out_corridor.assignment_weights[:1],
    )
    with pytest.raises(ValueError, match="held-out corridor does not cover the split"):
        replace(expected, held_out_corridor=partial)


def test_p5_8_rejects_held_out_exemplar_subset_at_p5_4_materialization():
    values = _evaluation_values()
    expected = values["expected_batches"]
    partial = replace(
        expected.held_out_exemplars,
        input_ids=expected.held_out_exemplars.input_ids[:1],
        attention_mask=expected.held_out_exemplars.attention_mask[:1],
        example_ids=expected.held_out_exemplars.example_ids[:1],
        selected_example_indices=expected.held_out_exemplars.selected_example_indices[
            :1
        ],
        selected_positions=expected.held_out_exemplars.selected_positions[:1],
        sparse_targets=expected.held_out_exemplars.sparse_targets[:1],
        passports=expected.held_out_exemplars.passports[:1],
    )
    with pytest.raises(
        ValueError, match="exemplar batches do not cover policy assignments"
    ):
        replace(expected, held_out_exemplars=partial)


def test_p5_4_rejects_raw_authoritative_passport_override():
    with pytest.raises(TypeError, match="authoritative_exemplar_passport_keys"):
        materialize_behavioral_batches_v1(
            _projection(), authoritative_exemplar_passport_keys=()
        )
