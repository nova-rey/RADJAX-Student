"""M2.4 pure-JAX recurrence, chunk, and gradient evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# JAX is optional for the normal static suite.
# ruff: noqa: E402, I001
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from radjax_student.architecture import ArchitectureInitRequest
from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.mamba2_reference import (
    Mamba2ReferencePlugin,
    reference_architecture_config,
)
from radjax_student.architecture.mamba2_reference.kernels import (
    mamba2_sequence,
    mamba2_step,
)
from radjax_student.contracts import ObjectiveScope
from radjax_student.learning.jax_core import JaxBatch
from radjax_student.runtime.jax_bridge import materialize_initialization_jax_key


_ORACLE_FIXTURE = (
    Path(__file__).parents[1] / "evidence/mamba2_oracle/full_token_step_fixture.json"
)
_CORE_WITNESS = Path(__file__).parents[1] / "evidence/mamba2_oracle/witness.json"


def _reject_nonfinite_json_constants(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _initialized():
    config = reference_architecture_config()
    reference = "runtime_keys.v1:initialization:42"
    request = ArchitectureInitRequest(
        config,
        reference,
        "float32",
        runtime_initialization_material=materialize_initialization_jax_key(reference),
    )
    return config, Mamba2ReferencePlugin().initialize_parameters(request)


def _tokens(values=(1, 7, 3, 12)):
    return jnp.asarray([values], dtype=jnp.int32)


def _tree_from_dotted_state(state):
    tree = {}
    for dotted, value in state.items():
        cursor = tree
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = jnp.asarray(value, dtype=jnp.float32)
    return tree


@pytest.mark.jax
def test_pinned_upstream_token_step_fixture_matches_logits_and_state():
    fixture = json.loads(
        _ORACLE_FIXTURE.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json_constants,
    )
    assert fixture["config"]["ssm_cfg"]["dt_limit"] == {
        "min": 0.0,
        "max": "UNBOUNDED",
    }
    recorded_digest = fixture["fixture_sha256"]
    digest_payload = dict(fixture)
    digest_payload.pop("fixture_sha256")
    assert (
        hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == recorded_digest
    )
    parameters = _tree_from_dotted_state(fixture["model_state_dict"])
    carry = {}
    for layer, values in fixture["state_initial"].items():
        carry[f"layers.{layer}.conv_state"] = jnp.asarray(values[0], dtype=jnp.float32)
        carry[f"layers.{layer}.ssm_state"] = jnp.asarray(values[1], dtype=jnp.float32)
    logits, final_carry = mamba2_sequence(
        parameters, jnp.asarray(fixture["tokens"], dtype=jnp.int32), carry
    )
    np.testing.assert_allclose(
        np.asarray(logits), np.asarray(fixture["logits"]), atol=5e-6, rtol=5e-6
    )
    for layer, values in fixture["state_final"].items():
        np.testing.assert_allclose(
            np.asarray(final_carry[f"layers.{layer}.conv_state"]),
            np.asarray(values[0]),
            atol=5e-6,
            rtol=5e-6,
        )
        np.testing.assert_allclose(
            np.asarray(final_carry[f"layers.{layer}.ssm_state"]),
            np.asarray(values[1]),
            atol=5e-6,
            rtol=5e-6,
        )


@pytest.mark.jax
def test_independent_asymmetric_state_witness_exercises_both_state_axes():
    """The checked-in upstream core witness prevents zero-state-only coverage."""
    witness = json.loads(_CORE_WITNESS.read_text(encoding="utf-8"))
    case = witness["asymmetric_nonzero_state_case"]
    initial = case["initial_state"]
    assert any(value for layer in initial.values() for value in layer["conv"]["values"])
    assert any(value for layer in initial.values() for value in layer["ssm"]["values"])
    _, initialized = _initialized()
    carry = {}
    for layer, values in initial.items():
        carry[f"layers.{layer}.conv_state"] = jnp.asarray(
            values["conv"]["values"], dtype=jnp.float32
        ).reshape(values["conv"]["shape"])
        carry[f"layers.{layer}.ssm_state"] = jnp.asarray(
            values["ssm"]["values"], dtype=jnp.float32
        ).reshape(values["ssm"]["shape"])
    token = jnp.asarray(int(case["tokens"][0][0]), dtype=jnp.int32)
    asymmetric_logits, asymmetric_carry = mamba2_step(
        initialized.parameters, token, carry
    )
    zero_logits, zero_carry = mamba2_step(
        initialized.parameters, token, initialized.architecture_carry
    )
    assert not bool(jnp.allclose(asymmetric_logits, zero_logits))
    assert any(
        not bool(jnp.allclose(asymmetric_carry[name], zero_carry[name]))
        for name in asymmetric_carry
    )


@pytest.mark.jax
def test_step_and_sequence_agree_on_logits_and_both_state_families():
    config, result = _initialized()
    tokens = _tokens()
    sequence_logits, sequence_carry = mamba2_sequence(
        result.parameters, tokens[0], result.architecture_carry
    )
    step_logits = []
    carry = result.architecture_carry
    for token in tokens[0]:
        logits, carry = mamba2_step(result.parameters, token, carry)
        step_logits.append(logits[0])
    np.testing.assert_allclose(
        np.asarray(sequence_logits), np.asarray(step_logits), atol=1e-6, rtol=1e-6
    )
    for name in sequence_carry:
        np.testing.assert_allclose(
            np.asarray(sequence_carry[name]),
            np.asarray(carry[name]),
            atol=1e-6,
            rtol=1e-6,
        )
    output = Mamba2ReferencePlugin().apply_jax(
        result.parameters,
        result.architecture_carry,
        JaxBatch(inputs={"token_ids": tokens}, targets={}),
        architecture_config=config,
        objective_scope=ObjectiveScope(),
        training=False,
        rng_key=None,
    )
    assert output.outputs.shape == (1, 4, 16)


@pytest.mark.jax
def test_fixed_weight_chunked_continuation_matches_whole_sequence():
    _, result = _initialized()
    tokens = jnp.asarray([1, 7, 3, 12], dtype=jnp.int32)
    whole_logits, whole_carry = mamba2_sequence(
        result.parameters, tokens, result.architecture_carry
    )
    first_logits, first_carry = mamba2_sequence(
        result.parameters, tokens[:2], result.architecture_carry
    )
    second_logits, second_carry = mamba2_sequence(
        result.parameters, tokens[2:], first_carry
    )
    np.testing.assert_allclose(
        np.asarray(whole_logits),
        np.asarray(jnp.concatenate((first_logits, second_logits))),
        atol=0,
        rtol=0,
    )
    for name in whole_carry:
        np.testing.assert_allclose(
            np.asarray(whole_carry[name]),
            np.asarray(second_carry[name]),
            atol=0,
            rtol=0,
        )


@pytest.mark.jax
def test_representative_gradient_families_are_finite_and_informative():
    _, result = _initialized()
    tokens = jnp.asarray([1, 7, 3, 12], dtype=jnp.int32)

    def loss(parameters):
        logits, _ = mamba2_sequence(parameters, tokens, result.architecture_carry)
        return jnp.mean(logits * jnp.asarray(3.7e11, dtype=jnp.float32))

    gradients = jax.grad(loss)(result.parameters)
    paths = {
        "convolution": ("backbone", "layers", "0", "mixer", "conv1d", "weight"),
        "input_projection": ("backbone", "layers", "0", "mixer", "in_proj", "weight"),
        "A_log": ("backbone", "layers", "0", "mixer", "A_log"),
        "dt_bias": ("backbone", "layers", "0", "mixer", "dt_bias"),
        "normalization": ("backbone", "layers", "0", "mixer", "norm", "weight"),
        "output_projection": ("backbone", "layers", "0", "mixer", "out_proj", "weight"),
    }
    for family, path in paths.items():
        value = gradients
        for key in path:
            value = value[key]
        assert bool(jnp.all(jnp.isfinite(value))), family
        assert float(jnp.max(jnp.abs(value))) > 1e-12, family


@pytest.mark.jax
def test_gradient_probe_rejects_zeroed_or_sign_reversed_family():
    _, result = _initialized()
    tokens = jnp.asarray([1, 7, 3, 12], dtype=jnp.int32)

    def loss(parameters):
        logits, _ = mamba2_sequence(parameters, tokens, result.architecture_carry)
        return jnp.mean(logits)

    gradients = jax.grad(loss)(result.parameters)
    path = ("backbone", "layers", "0", "mixer", "in_proj", "weight")
    expected = gradients
    for key in path:
        expected = expected[key]
    assert float(jnp.max(jnp.abs(expected))) > 1e-12
    zeroed = jnp.zeros_like(expected)
    reversed_gradient = -expected
    assert not bool(jnp.allclose(zeroed, expected, atol=1e-12, rtol=0))
    assert not bool(jnp.allclose(reversed_gradient, expected, atol=1e-12, rtol=0))


@pytest.mark.jax
def test_independent_finite_difference_gradient_reference_covers_state_parameters():
    """Central differences check the functional state transition independently."""
    _, initialized = _initialized()
    parameters = copy.deepcopy(initialized.parameters)
    for layer in parameters["backbone"]["layers"].values():
        layer["norm"]["weight"] = jnp.ones_like(layer["norm"]["weight"])
        layer["mixer"]["norm"]["weight"] = jnp.ones_like(
            layer["mixer"]["norm"]["weight"]
        )
        layer["mixer"]["D"] = jnp.ones_like(layer["mixer"]["D"])
        layer["mixer"]["dt_bias"] = jnp.full_like(layer["mixer"]["dt_bias"], 3.0)
        layer["mixer"]["A_log"] = jnp.zeros_like(layer["mixer"]["A_log"])
        layer["mixer"]["conv1d"]["bias"] = jnp.ones_like(
            layer["mixer"]["conv1d"]["bias"]
        )
    parameters["backbone"]["norm_f"]["weight"] = jnp.ones_like(
        parameters["backbone"]["norm_f"]["weight"]
    )
    tokens = jnp.asarray([1, 7, 3, 12], dtype=jnp.int32)

    def loss(candidate):
        logits, _ = mamba2_sequence(candidate, tokens, initialized.architecture_carry)
        return jnp.mean(logits * jnp.asarray(1.0e4, dtype=jnp.float32))

    gradients = jax.grad(loss)(parameters)
    paths = {
        "convolution": ("backbone", "layers", "0", "mixer", "conv1d", "weight"),
        "input_projection": (
            "backbone",
            "layers",
            "0",
            "mixer",
            "in_proj",
            "weight",
        ),
        "A_log": ("backbone", "layers", "0", "mixer", "A_log"),
        "dt_bias": ("backbone", "layers", "0", "mixer", "dt_bias"),
        "normalization": ("backbone", "layers", "0", "mixer", "norm", "weight"),
        "output_projection": (
            "backbone",
            "layers",
            "0",
            "mixer",
            "out_proj",
            "weight",
        ),
    }
    for family, path in paths.items():
        leaf = parameters
        derivative = gradients
        for key in path:
            leaf = leaf[key]
            derivative = derivative[key]
        index = np.unravel_index(int(jnp.argmax(jnp.abs(derivative))), leaf.shape)
        plus = copy.deepcopy(parameters)
        minus = copy.deepcopy(parameters)
        plus_cursor = plus
        minus_cursor = minus
        for key in path[:-1]:
            plus_cursor = plus_cursor[key]
            minus_cursor = minus_cursor[key]
        epsilon = 1.0e-3
        plus_cursor[path[-1]] = leaf.at[index].add(epsilon)
        minus_cursor[path[-1]] = leaf.at[index].add(-epsilon)
        finite_difference = (float(loss(plus)) - float(loss(minus))) / (2 * epsilon)
        np.testing.assert_allclose(
            finite_difference,
            float(derivative[index]),
            atol=0.2,
            rtol=0.03,
            err_msg=family,
        )


@pytest.mark.jax
def test_malformed_tokens_and_carry_fail_closed():
    config, result = _initialized()
    plugin = Mamba2ReferencePlugin()
    with pytest.raises(ArchitectureContractError):
        plugin.apply_jax(
            result.parameters,
            result.architecture_carry,
            JaxBatch(inputs={"token_ids": jnp.asarray([[1] * 9])}, targets={}),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
    bad_carry = dict(result.architecture_carry)
    bad_carry["layers.0.ssm_state"] = jnp.zeros((1, 4, 4, 5), dtype=jnp.float32)
    with pytest.raises(ArchitectureContractError):
        plugin.apply_jax(
            result.parameters,
            bad_carry,
            JaxBatch(inputs={"token_ids": _tokens()}, targets={}),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
