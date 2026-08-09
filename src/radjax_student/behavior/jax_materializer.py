"""Production JAX materialization for sealed neutral behavior source units."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from radjax_student.behavior.learning_batch import (
    CORRIDOR_SOURCE_UNIT_KIND_V1,
    EXEMPLAR_SOURCE_UNIT_KIND_V1,
    BehavioralLearningBatchError,
    validate_behavioral_source_unit_learning_batch_v1,
)
from radjax_student.learning import LearningBatch
from radjax_student.learning.jax_batch import (
    JaxBatchMaterializer,
    learning_batch_digest,
)


class BehaviorJaxBatchMaterializerV1:
    """Convert one verified neutral behavior source unit to JAX values.

    This materializer owns no source residency and no model concern.  Its
    input is a self-contained finite-JSON ``LearningBatch`` made by the P6.2
    source-unit factories; validation is deliberately repeated before every
    JAX conversion so altered metadata cannot select different source values.
    """

    materializer_id = "behavior_jax_batch_materializer.v1"

    def materialize(self, batch: LearningBatch) -> Any:
        source = validate_behavioral_source_unit_learning_batch_v1(batch)
        jnp = import_module("jax.numpy")
        from radjax_student.learning.jax_core import JaxBatch

        inputs = {
            "token_ids": jnp.asarray(batch.inputs["token_ids"], dtype=jnp.int32),
            "attention_mask": jnp.asarray(
                batch.inputs["attention_mask"], dtype=jnp.int32
            ),
        }
        if source.kind == CORRIDOR_SOURCE_UNIT_KIND_V1:
            targets = _corridor_targets(jnp, batch.targets)
        elif source.kind == EXEMPLAR_SOURCE_UNIT_KIND_V1:
            targets = _exemplar_targets(jnp, batch.targets)
        else:  # pragma: no cover - source-unit validation has already closed this.
            raise BehavioralLearningBatchError("behavior source-unit kind is invalid")
        return JaxBatch(
            inputs=inputs,
            targets=targets,
            weights=(
                {"attention_mask": inputs["attention_mask"]}
                if source.kind == CORRIDOR_SOURCE_UNIT_KIND_V1
                else None
            ),
            source_batch_digest=learning_batch_digest(batch),
        )


def _corridor_targets(jnp: Any, targets: Any) -> dict[str, Any]:
    mode_id = int(targets["mode_id"][0])
    statistics = targets["mode_bounds"].get(str(mode_id))
    if not isinstance(statistics, Mapping):
        raise BehavioralLearningBatchError("corridor source-unit bounds are invalid")
    names = ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass")
    try:
        lower = tuple(statistics[name][0] for name in names)
        upper = tuple(statistics[name][2] for name in names)
    except (KeyError, IndexError, TypeError) as exc:
        raise BehavioralLearningBatchError(
            "corridor source-unit bounds are invalid"
        ) from exc
    return {
        "position": jnp.asarray(targets["position"], dtype=jnp.int32),
        "mode_id": jnp.asarray(targets["mode_id"], dtype=jnp.int32),
        "assignment_weight": jnp.asarray(
            targets["assignment_weight"], dtype=jnp.float32
        ),
        "statistic_lower": jnp.asarray((lower,), dtype=jnp.float32),
        "statistic_upper": jnp.asarray((upper,), dtype=jnp.float32),
    }


def _exemplar_targets(jnp: Any, targets: Any) -> dict[str, Any]:
    return {
        "selected_position": jnp.asarray(targets["selected_position"], dtype=jnp.int32),
        "top_token_ids": jnp.asarray(targets["top_token_ids"], dtype=jnp.int32),
        "top_probabilities": jnp.asarray(
            targets["top_probabilities"], dtype=jnp.float32
        ),
        "aggregate_tail_mass": jnp.asarray(
            targets["aggregate_tail_mass"], dtype=jnp.float32
        ),
    }


BehavioralJaxBatchMaterializer = BehaviorJaxBatchMaterializerV1

assert isinstance(BehaviorJaxBatchMaterializerV1(), JaxBatchMaterializer)


__all__ = ["BehaviorJaxBatchMaterializerV1", "BehavioralJaxBatchMaterializer"]
