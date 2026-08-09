"""P6.2 neutral B=1 behavior source-unit materialization."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from radjax_student.behavior import (  # noqa: E402
    BehavioralJaxBatchMaterializer,
    BehavioralLearningBatchError,
    corridor_source_unit_learning_batch_v1,
    exemplar_source_unit_learning_batch_v1,
    materialize_behavioral_batches_v1,
    validate_behavioral_source_unit_learning_batch_v1,
)
from radjax_student.learning.jax_batch import learning_batch_digest  # noqa: E402
from tests.test_p5_4_behavior_materialization import _projection  # noqa: E402

pytestmark = pytest.mark.jax


def _batches():
    return materialize_behavioral_batches_v1(_projection())


def test_p6_2_factories_are_deterministic_b1_neutral_source_units():
    materialization = _batches()
    corridor = corridor_source_unit_learning_batch_v1(
        materialization,
        partition="training",
        coordinate=("corpus_000000001", 0, 0),
    )
    repeated = corridor_source_unit_learning_batch_v1(
        materialization,
        partition="training",
        coordinate=("corpus_000000001", 0, 0),
    )
    exemplar = exemplar_source_unit_learning_batch_v1(
        materialization,
        partition="training",
        passport_key=("corpus_000000001", 2, "a"),
    )

    assert corridor == repeated
    assert corridor.inputs["token_ids"] == ((0, 1, 2, 3),)
    assert exemplar.inputs["token_ids"] == ((0, 1, 2, 3),)
    assert corridor.metadata["coordinate"] == {
        "example_id": "corpus_000000001",
        "position": 0,
        "mode_id": 0,
    }
    assert exemplar.metadata["coordinate"] == {
        "example_id": "corpus_000000001",
        "position": 2,
        "corridor_fingerprint_id": "a",
    }
    for batch in (corridor, exemplar):
        assert set(batch.metadata).isdisjoint(
            {"artifact", "checkpoint", "architecture", "parameters"}
        )
        assert validate_behavioral_source_unit_learning_batch_v1(batch).partition == (
            "training"
        )


def test_p6_2_materializer_preserves_learning_digest_and_neutral_values():
    materialization = _batches()
    corridor = corridor_source_unit_learning_batch_v1(
        materialization,
        partition="training",
        coordinate=("corpus_000000001", 0, 0),
    )
    exemplar = exemplar_source_unit_learning_batch_v1(
        materialization,
        partition="training",
        passport_key=("corpus_000000001", 2, "a"),
    )

    materializer = BehavioralJaxBatchMaterializer()
    corridor_jax = materializer.materialize(corridor)
    exemplar_jax = materializer.materialize(exemplar)

    for source, materialized in ((corridor, corridor_jax), (exemplar, exemplar_jax)):
        assert materialized.source_batch_digest == learning_batch_digest(source)
        assert materialized.inputs["token_ids"].shape == (1, 4)
        assert materialized.inputs["attention_mask"].shape == (1, 4)
    assert np.array_equal(corridor_jax.targets["position"], np.asarray([0]))
    assert np.array_equal(exemplar_jax.targets["top_token_ids"], np.asarray([[1, 2]]))


@pytest.mark.parametrize(
    "alteration",
    (
        lambda batch: replace(
            batch,
            metadata={**batch.metadata, "partition": "held_out"},
        ),
        lambda batch: replace(
            batch,
            metadata={**batch.metadata, "coordinate": {"example_id": "forged"}},
        ),
        lambda batch: replace(
            batch,
            inputs={"token_ids": batch.inputs["token_ids"], "attention_mask": ((1,),)},
        ),
        lambda batch: replace(
            batch,
            targets={**batch.targets, "position": (1,)},
        ),
    ),
)
def test_p6_2_materializer_rejects_malformed_or_forged_source_metadata(alteration):
    batch = corridor_source_unit_learning_batch_v1(
        _batches(),
        partition="training",
        coordinate=("corpus_000000001", 0, 0),
    )

    with pytest.raises(BehavioralLearningBatchError):
        BehavioralJaxBatchMaterializer().materialize(alteration(batch))
