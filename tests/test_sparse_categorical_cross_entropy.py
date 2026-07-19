"""Focused P4.5 contracts for the generic sparse categorical objective."""

from __future__ import annotations

# JAX availability is checked before importing JAX-bearing production modules.
# ruff: noqa: E402
import math

import pytest

jax = pytest.importorskip(
    "jax", reason="Sparse categorical cross-entropy tests require JAX"
)
jnp = pytest.importorskip(
    "jax.numpy", reason="Sparse categorical cross-entropy tests require JAX"
)

from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveScope,
    ResolvedObjectiveSelection,
)
from radjax_student.objectives import (
    SPARSE_CROSS_ENTROPY_IDENTITY,
    SparseCategoricalCrossEntropyObjective,
    build_default_objective_registry,
)

pytestmark = pytest.mark.jax


def _config() -> ObjectiveConfig:
    return ObjectiveConfig(SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"})


def _evaluate(logits, target, *, weights=None):
    return SparseCategoricalCrossEntropyObjective().evaluate_jax(
        surface=logits,
        targets={"token_ids": target},
        weights=weights,
        config=_config(),
    )


def test_identity_profile_registry_and_logits_surface_are_generic() -> None:
    objective = SparseCategoricalCrossEntropyObjective()
    profile = objective.capability_profile()
    selection = build_default_objective_registry().select(SPARSE_CROSS_ENTROPY_IDENTITY)

    assert selection.plugin.objective_identity() == SPARSE_CROSS_ENTROPY_IDENTITY
    assert profile.required_surface_roles == ("logits",)
    assert profile.target_requirements == ("targets.token_ids",)
    objective.validate_resolved_surface(
        ResolvedObjectiveSelection(ObjectiveScope(), "final_output", "logits")
    )


def test_mean_nll_and_accuracy_match_a_small_independent_calculation() -> None:
    logits = jnp.asarray([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
    target = jnp.asarray([[0, 2]], dtype=jnp.int32)

    loss, metrics = _evaluate(logits, target)
    first_nll = -(2.0 - math.log(math.exp(2.0) + 1.0 + math.exp(-1.0)))
    second_nll = -(0.0 - math.log(math.exp(1.0) + 2.0))
    expected = (first_nll + second_nll) / 2.0

    assert float(loss) == pytest.approx(expected)
    assert float(metrics["objective.sparse_cross_entropy"]) == pytest.approx(expected)
    assert float(metrics["objective.token_accuracy"]) == pytest.approx(0.5)
    SparseCategoricalCrossEntropyObjective().validate_metrics(metrics)


def test_valid_jit_execution_matches_eager() -> None:
    logits = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    target = jnp.asarray([[0, 1]], dtype=jnp.int32)
    eager = _evaluate(logits, target)
    compiled = jax.jit(_evaluate)(logits, target)

    assert jnp.allclose(eager[0], compiled[0])
    assert jnp.allclose(
        eager[1]["objective.token_accuracy"],
        compiled[1]["objective.token_accuracy"],
    )


def test_rejects_bad_config_targets_surface_weights_and_eager_range() -> None:
    objective = SparseCategoricalCrossEntropyObjective()
    logits = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    valid_target = jnp.asarray([[0, 1]], dtype=jnp.int32)
    invalid_cases = (
        (ObjectiveConfig(SPARSE_CROSS_ENTROPY_IDENTITY, {}), None),
        (_config(), {"token_ids": jnp.asarray([[0.0, 1.0]])}),
        (_config(), {"token_ids": jnp.asarray([[0, -1]], dtype=jnp.int32)}),
    )
    for config, targets in invalid_cases:
        with pytest.raises(ObjectiveContractError):
            if targets is None:
                objective.validate_config(config)
            else:
                objective.validate_targets(targets)

    with pytest.raises(ObjectiveContractError, match="within the logits vocabulary"):
        _evaluate(logits, jnp.asarray([[0, 3]], dtype=jnp.int32))
    with pytest.raises(ObjectiveContractError, match="does not support"):
        _evaluate(logits, valid_target, weights={"token_ids": jnp.ones((1, 2))})
    with pytest.raises(ObjectiveContractError, match="logits"):
        _evaluate(jnp.zeros((1, 2)), valid_target)


def test_traced_out_of_range_target_cannot_be_silently_clipped() -> None:
    logits = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    target = jnp.asarray([[0, 3]], dtype=jnp.int32)
    loss, metrics = jax.jit(_evaluate)(logits, target)

    assert not bool(jnp.isfinite(loss))
    assert not bool(jnp.isfinite(metrics["objective.token_accuracy"]))
