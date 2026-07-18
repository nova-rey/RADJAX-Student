"""JAX-free runtime callable identity contract tests."""

from __future__ import annotations

import pytest

from radjax_student.runtime.callables import (
    CALLABLE_DECLARATION_SCHEMA_VERSION,
    RuntimeCallableDeclaration,
    RuntimeCallableError,
    RuntimeCallableReference,
    final_prepared_execution_identity,
)
from radjax_student.runtime.execution import PreparedExecutionIdentityCache
from radjax_student.validation.p3_12d_runtime_callable_identity.models import (
    AdversarialResult,
    PositiveResult,
)


@pytest.mark.jax
def test_default_generic_step_binding_is_source_derived_and_deterministic() -> None:
    pytest.importorskip("jax")
    registry = _default_registry()
    first = registry.resolve(_reference_from_default_registry())
    second = registry.resolve(first.reference)
    assert first.identity == second.identity
    assert first.reference.callable_id == "radjax.learning.generic_jax_step"
    assert (
        first.identity.implementation_source_digest != first.identity.declaration_digest
    )
    assert first.to_dict().keys() == {"declaration", "identity"}


@pytest.mark.jax
def test_reference_rejects_unknown_fields_and_wrong_identity() -> None:
    reference = RuntimeCallableReference(
        "radjax.runtime_callable_reference.v1",
        "radjax.learning.generic_jax_step",
        1,
        "1" * 64,
    )
    payload = reference.to_dict() | {"unknown": True}
    with pytest.raises(ValueError, match="unknown"):
        RuntimeCallableReference.from_dict(payload)
    pytest.importorskip("jax")
    registry = _default_registry()
    with pytest.raises(RuntimeCallableError, match="request_mismatch"):
        registry.resolve(
            RuntimeCallableReference(
                reference.schema_version,
                reference.callable_id,
                reference.callable_version,
                "0" * 64,
            )
        )


def test_declaration_is_strict_and_rejects_validation_owner() -> None:
    with pytest.raises(ValueError, match="production-owned"):
        RuntimeCallableDeclaration(
            CALLABLE_DECLARATION_SCHEMA_VERSION,
            "bad",
            1,
            "validation",
            "radjax_student.validation.bad",
            "bad",
            "input",
            "output",
        )


def test_declaration_round_trip_rejects_unknown_fields() -> None:
    declaration = RuntimeCallableDeclaration(
        CALLABLE_DECLARATION_SCHEMA_VERSION,
        "radjax.runtime.test_operation",
        1,
        "runtime",
        "radjax_student.runtime.portability",
        "_scale_add",
        "input.v1",
        "output.v1",
    )
    assert RuntimeCallableDeclaration.from_dict(declaration.to_dict()) == declaration
    with pytest.raises(ValueError, match="unknown"):
        RuntimeCallableDeclaration.from_dict(declaration.to_dict() | {"extra": True})


def test_prepared_identity_binds_both_static_contract_and_actual_values() -> None:
    reference = RuntimeCallableReference(
        "radjax.runtime_callable_reference.v1",
        "radjax.runtime.test_operation",
        1,
        "1" * 64,
    )
    common = {
        "reference": reference,
        "backend_id": "runtime.test",
        "runtime_id": "runtime.test.context",
        "runtime_implementation_version": "1",
        "mode": "jit",
        "compilation_options": {"enabled": True, "mode": "jit"},
        "input_signature": {"args": ["scalar"]},
        "donation_contract": {"names": [], "positions": []},
        "placement_plan_identity": "single-device",
        "required_capabilities": ("compilation.jit_v1",),
    }
    first = final_prepared_execution_identity(
        **common,
        static_contract={"positions": [0], "names": []},
        static_values={"position:0": 1},
    )
    changed_value = final_prepared_execution_identity(
        **common,
        static_contract={"positions": [0], "names": []},
        static_values={"position:0": 2},
    )
    changed_contract = final_prepared_execution_identity(
        **common,
        static_contract={"positions": [], "names": ["scale"]},
        static_values={"name:scale": 1},
    )
    assert (
        first.static_argument_contract_digest
        != changed_contract.static_argument_contract_digest
    )
    assert (
        first.static_argument_value_digest != changed_value.static_argument_value_digest
    )
    assert first.prepared_execution_digest != changed_value.prepared_execution_digest
    assert first.prepared_execution_digest != changed_contract.prepared_execution_digest


def test_cache_reuses_only_exact_prepared_identity() -> None:
    reference = RuntimeCallableReference(
        "radjax.runtime_callable_reference.v1",
        "radjax.runtime.test_operation",
        1,
        "1" * 64,
    )
    identity = final_prepared_execution_identity(
        reference=reference,
        backend_id="runtime.test",
        runtime_id="runtime.test.context",
        runtime_implementation_version="1",
        mode="eager",
        compilation_options={"enabled": False, "mode": "eager"},
        input_signature={"argument_count": 1},
        static_contract={"names": [], "positions": []},
        static_values={},
        donation_contract={"names": [], "positions": []},
        placement_plan_identity="single-device",
        required_capabilities=("execution.eager_v1",),
    )
    cache = PreparedExecutionIdentityCache()
    assert cache.record(identity, cache_policy="reuse") is False
    assert cache.record(identity, cache_policy="reuse") is True
    with pytest.raises(Exception, match="cache_identity_mismatch"):
        cache.record(identity, cache_policy="disabled")


def test_typed_evidence_records_round_trip_strictly() -> None:
    positive = PositiveResult("callable_registered", "runtime.registry", "1" * 64)
    assert PositiveResult.from_dict(positive.to_dict()) == positive
    with pytest.raises(ValueError, match="unknown"):
        PositiveResult.from_dict(positive.to_dict() | {"passed": True})
    adversarial = AdversarialResult(
        case_id="wrong_declaration_type",
        category="declaration",
        mutation_applied=True,
        baseline_input_digest="1" * 64,
        mutated_input_digest="2" * 64,
        intended_boundary="runtime.bind",
        boundary_callable_identity="runtime.bind",
        observed_boundary="runtime.bind",
        observed_exception_type="RuntimeCallableError",
        expected_code="execution_callable_declaration_invalid",
        observed_code="execution_callable_declaration_invalid",
        deterministic_first_failure=True,
        first_run_evidence_digest="3" * 64,
        second_run_evidence_digest="3" * 64,
        outcome="reject",
    )
    assert AdversarialResult.from_dict(adversarial.to_dict()) == adversarial
    with pytest.raises(ValueError, match="unknown"):
        AdversarialResult.from_dict(adversarial.to_dict() | {"accepted": True})


def _reference_from_default_registry() -> RuntimeCallableReference:
    registry = _default_registry()
    # The API deliberately has no discovery: this reference is taken from a
    # known declared operation, then resolution remains exact.
    binding = next(iter(registry._bindings.values()))
    return binding.reference


def _default_registry():
    pytest.importorskip("jax")
    from radjax_student.learning.composition import (
        build_default_learning_callable_registry,
    )

    return build_default_learning_callable_registry()
