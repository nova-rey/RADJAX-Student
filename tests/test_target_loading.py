from pathlib import Path

import numpy as np
from radjax_contract.tome import TomeManifest, TomeShard
from radjax_contract.vocab import VocabContract
from tome_fixtures import write_dense_tome

from radjax_student.artifacts.targets import (
    load_dense_tome_targets,
    target_batch_from_dense_tome,
)


def test_dense_tome_targets_load_logits_records_and_probabilities(
    tmp_path: Path,
) -> None:
    tome = tmp_path / "tome"
    logits = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.0, 0.5, 1.0]],
            [[3.0, 2.0, 1.0], [1.0, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )
    write_dense_tome(tome, logits=logits)

    targets = load_dense_tome_targets(tome)
    batch = target_batch_from_dense_tome(targets)

    assert targets.logits.shape == (2, 2, 3)
    assert tuple(record.example_id for record in targets.records) == (
        "example-0",
        "example-1",
    )
    assert np.allclose(targets.probabilities().sum(axis=-1), 1.0)
    assert batch["payload_format"] == "dense_logits_v0"


def test_dense_tome_targets_concatenate_manifest_shards(tmp_path: Path) -> None:
    tome = tmp_path / "tome"
    write_dense_tome(
        tome,
        manifest=(
            TomeManifest(
                vocab_contract=VocabContract(tokenizer_id="toy", vocab_size=3),
                record_count=2,
                sequence_length=2,
                shard_count=2,
                shards=(
                    TomeShard(path="shard-0.npy", record_count=1),
                    TomeShard(path="shard-1.npy", record_count=1),
                ),
            )
        ),
    )
    np.save(tome / "shard-0.npy", np.ones((1, 2, 3), dtype=np.float32))
    np.save(tome / "shard-1.npy", np.zeros((1, 2, 3), dtype=np.float32))

    targets = load_dense_tome_targets(tome)

    assert targets.logits.shape == (2, 2, 3)
    assert np.all(targets.logits[0] == 1.0)
    assert np.all(targets.logits[1] == 0.0)
