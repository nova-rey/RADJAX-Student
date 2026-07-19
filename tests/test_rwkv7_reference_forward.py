"""P4.4 fixture-domain forward parity for the RWKV-7 reference plugin."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from radjax_student.architecture import (
    ArchitectureContractError,
    ArchitectureRegistry,
    JaxArchitecturePlugin,
)
from radjax_student.architecture.rwkv7_reference import register_rwkv7_reference
from radjax_student.architecture.rwkv7_reference.kernels import (
    rwkv7_sequence,
    rwkv7_step,
)
from radjax_student.architecture.rwkv7_reference.plugin import RWKV7ReferencePlugin
from radjax_student.contracts import ObjectiveScope
from radjax_student.learning.jax_core import JaxBatch
from tests.support.generate_rwkv7_reference_fixture import (
    fixture_bytes,
    provenance_bytes,
)
from tests.support.rwkv7_reference_oracle import (
    fixture_carry,
    fixture_parameters,
)
from tests.support.rwkv7_reference_oracle import (
    rwkv7_sequence as oracle_sequence,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/rwkv7_reference/parity_fixture.json"
PROVENANCE_PATH = ROOT / "tests/fixtures/rwkv7_reference/provenance.json"
RTOL = 1e-5
ATOL = 2e-5


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _execution_values():
    fixture = _fixture()
    parameters = jax.tree_util.tree_map(jnp.asarray, fixture_parameters())
    carry = jax.tree_util.tree_map(jnp.asarray, fixture_carry())
    tokens = jnp.asarray(fixture["tokens"], dtype=jnp.int32)
    return fixture, parameters, carry, tokens


def _apply(parameters, carry, tokens):
    return RWKV7ReferencePlugin().apply_jax(
        parameters,
        carry,
        JaxBatch(inputs={"token_ids": tokens[None, :]}, targets={}),
        objective_scope=ObjectiveScope(),
        training=False,
        rng_key=None,
    )


def test_fixture_generator_oracle_and_provenance_are_deterministic() -> None:
    fixture_bytes_on_disk = FIXTURE_PATH.read_bytes()
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert fixture_bytes_on_disk == fixture_bytes()
    assert PROVENANCE_PATH.read_bytes() == provenance_bytes(fixture_bytes_on_disk)
    assert (
        provenance["fixture"]["sha256"]
        == hashlib.sha256(fixture_bytes_on_disk).hexdigest()
    )
    assert provenance["pinned_source"] == _fixture()["pinned_source"]

    expected_logits, expected_carry = oracle_sequence(
        fixture_parameters(), np.asarray(_fixture()["tokens"]), fixture_carry()
    )
    np.testing.assert_array_equal(
        expected_logits, np.asarray(_fixture()["expected_logits"], dtype=np.float32)
    )
    for name, expected in expected_carry.items():
        np.testing.assert_array_equal(
            expected, np.asarray(_fixture()["expected_carry"][name], dtype=np.float32)
        )


def test_plugin_sequence_matches_independent_pinned_fixture_and_carry() -> None:
    fixture, parameters, carry, tokens = _execution_values()
    result = _apply(parameters, carry, tokens)

    np.testing.assert_allclose(
        np.asarray(result.outputs)[0],
        np.asarray(fixture["expected_logits"], dtype=np.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    assert set(result.updated_architecture_carry) == {
        "last_x_time",
        "last_x_channel",
        "time_state_matrix",
    }
    for name, expected in fixture["expected_carry"].items():
        np.testing.assert_allclose(
            np.asarray(result.updated_architecture_carry[name]),
            np.asarray(expected, dtype=np.float32),
            rtol=RTOL,
            atol=ATOL,
        )


def test_stepwise_and_scan_sequence_agree_and_remain_finite() -> None:
    _, parameters, carry, tokens = _execution_values()
    sequence_logits, sequence_carry = rwkv7_sequence(parameters, tokens, carry)
    step_logits = []
    step_carry = carry
    for token in tokens:
        logits, step_carry = rwkv7_step(parameters, token, step_carry)
        step_logits.append(logits)

    np.testing.assert_allclose(
        np.asarray(jnp.stack(step_logits)),
        np.asarray(sequence_logits),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isfinite(np.asarray(sequence_logits)).all()
    for name in sequence_carry:
        np.testing.assert_allclose(
            np.asarray(step_carry[name]),
            np.asarray(sequence_carry[name]),
            rtol=RTOL,
            atol=ATOL,
        )
        assert np.isfinite(np.asarray(sequence_carry[name])).all()
        assert not np.array_equal(
            np.asarray(sequence_carry[name]), np.asarray(carry[name])
        )


def test_token_order_and_parameter_perturbation_change_the_execution() -> None:
    _, parameters, carry, tokens = _execution_values()
    baseline, _ = rwkv7_sequence(parameters, tokens, carry)
    reordered, _ = rwkv7_sequence(parameters, tokens[::-1], carry)
    perturbed = {
        **parameters,
        "head": {"weight": parameters["head"]["weight"] + jnp.float32(0.01)},
    }
    changed, _ = rwkv7_sequence(perturbed, tokens, carry)

    assert not np.allclose(np.asarray(baseline), np.asarray(reordered))
    assert not np.allclose(np.asarray(baseline), np.asarray(changed))


def test_execution_rejects_malformed_tokens_and_carry() -> None:
    _, parameters, carry, tokens = _execution_values()
    with pytest.raises(ValueError, match="rank one"):
        rwkv7_sequence(parameters, tokens[None, :], carry)
    for invalid in (
        JaxBatch(inputs={"token_ids": tokens.astype(jnp.float32)[None, :]}, targets={}),
        JaxBatch(inputs={"token_ids": tokens[:3][None, :]}, targets={}),
        JaxBatch(
            inputs={"token_ids": jnp.asarray([[1, 7, 3, 16]], dtype=jnp.int32)},
            targets={},
        ),
    ):
        with pytest.raises(ArchitectureContractError) as caught:
            RWKV7ReferencePlugin().apply_jax(
                parameters,
                carry,
                invalid,
                objective_scope=ObjectiveScope(),
                training=False,
                rng_key=None,
            )
        assert caught.value.code == "architecture_batch_incompatible"

    broken_carry = {**carry, "last_x_time": jnp.zeros((1, 8), dtype=jnp.float32)}
    with pytest.raises(ArchitectureContractError, match="persistent descriptor"):
        _apply(parameters, broken_carry, tokens)


def test_plugin_is_registered_jax_architecture_with_fixture_only_claim() -> None:
    registry = ArchitectureRegistry()
    plugin = register_rwkv7_reference(registry)

    assert isinstance(plugin, JaxArchitecturePlugin)
    assert plugin.capability_profile().supports("architecture.jax_execution_v1")
    assert registry.get(plugin.architecture_id) is plugin
    assert "equation_parity_outside_fixture_domain_not_claimed" in (
        plugin.architecture_metadata().claims_not_made
    )
