"""Literal P3.12D callable-identity adversaries.

Each experiment constructs one fresh mutation at the public owner boundary.
The runner deliberately owns neither the expected blocker nor observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial

from radjax_student.runtime.callables import (
    CALLABLE_DECLARATION_SCHEMA_VERSION,
    RuntimeCallableBinding,
    RuntimeCallableDeclaration,
    RuntimeCallableIdentity,
    RuntimeCallableReference,
    RuntimeCallableRegistry,
    bind_runtime_callable,
)
from radjax_student.runtime.execution import (
    ExecutionRequest,
    PreparedExecution,
    PreparedExecutionIdentityCache,
    execute_prepared,
    finalize_prepared_execution_identity,
    prepare_execution,
)
from radjax_student.runtime.models import (
    CompilationOptions,
    DeviceInventory,
    ExecutionContext,
    RuntimeCapabilityProfile,
    RuntimeEnvironment,
)
from radjax_student.runtime.portability import (
    PORTABILITY_SCALE_ADD_BINDING,
    _scale_add,
)

from .diagnostic import Invocation
from .implementation_audit import require_clean_synthetic_source


def _binding_input(declaration: object, callable_value: object) -> dict[str, object]:
    """Mechanical mutation evidence, independent of case metadata."""
    baseline = PORTABILITY_SCALE_ADD_BINDING
    return {
        "declaration": (
            declaration.to_dict()
            if hasattr(declaration, "to_dict")
            else type(declaration).__name__
        ),
        "callable_module": getattr(
            callable_value, "__module__", type(callable_value).__module__
        ),
        "callable_qualname": getattr(
            callable_value, "__qualname__", type(callable_value).__qualname__
        ),
        "is_baseline_callable": callable_value is _scale_add,
        "baseline_reference": baseline.reference.to_dict(),
    }


def _binding_invocation(declaration: object, callable_value: object) -> Invocation:
    return Invocation(
        bind_runtime_callable,
        (),
        {"callable": callable_value, "declaration": declaration},
        _binding_input(PORTABILITY_SCALE_ADD_BINDING.declaration, _scale_add),
        _binding_input(declaration, callable_value),
    )


def _registry() -> RuntimeCallableRegistry:
    registry = RuntimeCallableRegistry()
    registry.register(PORTABILITY_SCALE_ADD_BINDING)
    return registry


def _audit_invocation(source: str) -> Invocation:
    return Invocation(
        require_clean_synthetic_source,
        (source,),
        {},
        "clean P3.12D source",
        source,
    )


@dataclass
class _IdentityBackend:
    backend_id: str = "d-test"
    implementation_version: str = "1"

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            "d.test.capabilities.v1",
            self.backend_id,
            1,
            (
                "execution.eager_v1",
                "compilation.jit_v1",
                "execution.static_arguments_v1",
                "execution.argument_donation_v1",
            ),
        )

    def prepare_runtime_execution(self, context, function, request, mode):
        del context, request
        return {"function": function, "mode": mode}

    def compile_runtime_execution(self, context, handle, args, kwargs):
        del context, args, kwargs
        return handle, handle["mode"] == "jit"

    def dispatch_runtime_execution(self, context, handle, args, kwargs):
        del context
        return handle["function"](*args, **kwargs)

    def synchronize_runtime_execution(self, context, output):
        del context
        return output


def _execution_context(
    *, backend_id: str = "d-test", runtime_id: str = "d-test-runtime"
) -> ExecutionContext:
    return ExecutionContext(
        backend_id=backend_id,
        environment=RuntimeEnvironment(
            python_version="3.11",
            jax_available=False,
            process_count=1,
            process_index=0,
            local_device_count=1,
            global_device_count=1,
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            process_count=1, local_device_count=1, global_device_count=1
        ),
        capabilities=_IdentityBackend(backend_id).capability_profile(),
        root_seed=7,
        runtime_id=runtime_id,
    )


def _execution_request(
    *,
    mode: str = "eager",
    options: CompilationOptions | None = None,
    input_signature: dict[str, object] | None = None,
    placement_plan_id: str = "d-single-device",
) -> ExecutionRequest:
    resolved = CompilationOptions(mode=mode) if options is None else options
    return ExecutionRequest(
        request_id="p312d-runtime-request",
        function_id=PORTABILITY_SCALE_ADD_BINDING.reference.callable_id,
        mode=mode,
        compilation_options=resolved,
        placement_plan_id=placement_plan_id,
        input_signature=(
            {"argument_count": 1} if input_signature is None else input_signature
        ),
        callable_reference=PORTABILITY_SCALE_ADD_BINDING.reference,
    )


def _prepared_execution(
    *,
    mode: str = "eager",
    options: CompilationOptions | None = None,
    input_signature: dict[str, object] | None = None,
    placement_plan_id: str = "d-single-device",
) -> tuple[ExecutionContext, _IdentityBackend, PreparedExecution]:
    backend = _IdentityBackend()
    context = _execution_context()
    request = _execution_request(
        mode=mode,
        options=options,
        input_signature=input_signature,
        placement_plan_id=placement_plan_id,
    )
    return (
        context,
        backend,
        prepare_execution(
            context=context,
            callable_binding=PORTABILITY_SCALE_ADD_BINDING,
            request=request,
            backend=backend,
        ),
    )


def _finalized_prepared(
    *, options: CompilationOptions | None = None
) -> tuple[ExecutionContext, PreparedExecution]:
    context, _, prepared = _prepared_execution(options=options)
    identity = finalize_prepared_execution_identity(
        context=context, prepared=prepared, args=(1.0,)
    )
    return context, replace(prepared, prepared_identity=identity)


def experiment_wrong_declaration_type() -> Invocation:
    return _binding_invocation(object(), _scale_add)


def experiment_empty_callable_id() -> Invocation:
    return Invocation(
        RuntimeCallableDeclaration,
        (
            CALLABLE_DECLARATION_SCHEMA_VERSION,
            "",
            1,
            "runtime",
            "radjax_student.runtime.portability",
            "_scale_add",
            "input.v1",
            "output.v1",
        ),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.declaration.to_dict(),
        mutated_input={"callable_id": ""},
    )


def experiment_nonpositive_callable_version() -> Invocation:
    return Invocation(
        RuntimeCallableDeclaration,
        (
            CALLABLE_DECLARATION_SCHEMA_VERSION,
            "radjax.runtime.portability_scale_add",
            0,
            "runtime",
            "radjax_student.runtime.portability",
            "_scale_add",
            "input.v1",
            "output.v1",
        ),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.declaration.to_dict(),
        mutated_input={"callable_version": 0},
    )


def experiment_validation_owned_callable_rejected() -> Invocation:
    return Invocation(
        RuntimeCallableDeclaration,
        (
            CALLABLE_DECLARATION_SCHEMA_VERSION,
            "bad",
            1,
            "validation",
            "radjax_student.validation.fake",
            "fake",
            "input.v1",
            "output.v1",
        ),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.declaration.to_dict(),
        mutated_input={"implementation_module": "radjax_student.validation.fake"},
    )


def experiment_duplicate_callable_registration() -> Invocation:
    registry = _registry()
    return Invocation(
        registry.register,
        (PORTABILITY_SCALE_ADD_BINDING,),
        baseline_input={"registrations": 1},
        mutated_input={"registrations": 2},
    )


def experiment_conflicting_callable_registration() -> Invocation:
    registry = _registry()
    conflicting = bind_runtime_callable(
        callable=_scale_add,
        declaration=replace(PORTABILITY_SCALE_ADD_BINDING.declaration, owner="other"),
    )
    return Invocation(
        registry.register,
        (conflicting,),
        baseline_input={"owner": "runtime"},
        mutated_input={"owner": "other"},
    )


def experiment_unregistered_callable_reference() -> Invocation:
    registry = _registry()
    reference = RuntimeCallableReference(
        PORTABILITY_SCALE_ADD_BINDING.reference.schema_version,
        "radjax.runtime.missing",
        1,
        "0" * 64,
    )
    return Invocation(
        registry.resolve,
        (reference,),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.reference.to_dict(),
        mutated_input=reference.to_dict(),
    )


def experiment_lambda_callable_rejected() -> Invocation:
    return _binding_invocation(PORTABILITY_SCALE_ADD_BINDING.declaration, lambda x: x)


def experiment_nested_callable_rejected() -> Invocation:
    def nested(value: float) -> float:
        return value

    return _binding_invocation(PORTABILITY_SCALE_ADD_BINDING.declaration, nested)


def experiment_closure_callable_rejected() -> Invocation:
    factor = 2

    def closure(value: float) -> float:
        return value * factor

    return _binding_invocation(PORTABILITY_SCALE_ADD_BINDING.declaration, closure)


def experiment_partial_callable_rejected() -> Invocation:
    return _binding_invocation(
        PORTABILITY_SCALE_ADD_BINDING.declaration, partial(_scale_add, 1.0)
    )


def experiment_callable_instance_rejected() -> Invocation:
    class CallableInstance:
        def __call__(self, value: float) -> float:
            return value

    return _binding_invocation(
        PORTABILITY_SCALE_ADD_BINDING.declaration, CallableInstance()
    )


def experiment_bound_instance_method_rejected() -> Invocation:
    class BoundOwner:
        def method(self, value: float) -> float:
            return value

    return _binding_invocation(
        PORTABILITY_SCALE_ADD_BINDING.declaration, BoundOwner().method
    )


def experiment_dynamically_generated_callable_rejected() -> Invocation:
    namespace: dict[str, object] = {}
    exec("def generated(value):\n    return value\n", namespace)
    generated = namespace["generated"]
    generated.__module__ = "radjax_student.runtime.portability"
    generated.__name__ = "_scale_add"
    generated.__qualname__ = "_scale_add"
    return _binding_invocation(PORTABILITY_SCALE_ADD_BINDING.declaration, generated)


def experiment_correct_id_wrong_callable() -> Invocation:
    return _binding_invocation(PORTABILITY_SCALE_ADD_BINDING.declaration, _other_scale)


def experiment_correct_callable_wrong_id() -> Invocation:
    identity_payload = PORTABILITY_SCALE_ADD_BINDING.identity.to_dict()
    identity_payload["callable_id"] = "wrong.id"
    identity_payload["callable_identity_digest"] = hashlib.sha256(
        (
            json.dumps(
                {
                    key: value
                    for key, value in identity_payload.items()
                    if key not in {"schema_version", "callable_identity_digest"}
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    identity = RuntimeCallableIdentity(**identity_payload)
    return Invocation(
        RuntimeCallableBinding,
        (_scale_add, PORTABILITY_SCALE_ADD_BINDING.declaration, identity),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.identity.to_dict(),
        mutated_input=identity.to_dict(),
    )


def experiment_callable_version_mismatch() -> Invocation:
    registry = _registry()
    reference = replace(
        PORTABILITY_SCALE_ADD_BINDING.reference,
        callable_identity_digest="0" * 64,
    )
    return Invocation(
        registry.resolve,
        (reference,),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.reference.to_dict(),
        mutated_input=reference.to_dict(),
    )


def experiment_implementation_module_mismatch() -> Invocation:
    return _binding_invocation(
        replace(
            PORTABILITY_SCALE_ADD_BINDING.declaration,
            implementation_module="radjax_student.runtime.execution",
        ),
        _scale_add,
    )


def experiment_implementation_qualname_mismatch() -> Invocation:
    return _binding_invocation(
        replace(PORTABILITY_SCALE_ADD_BINDING.declaration, implementation_qualname="x"),
        _scale_add,
    )


def experiment_copied_identity_digest_rejected() -> Invocation:
    return Invocation(
        RuntimeCallableReference,
        (
            PORTABILITY_SCALE_ADD_BINDING.reference.schema_version,
            PORTABILITY_SCALE_ADD_BINDING.reference.callable_id,
            PORTABILITY_SCALE_ADD_BINDING.reference.callable_version,
            "not-a-digest",
        ),
        baseline_input=PORTABILITY_SCALE_ADD_BINDING.reference.to_dict(),
        mutated_input={"callable_identity_digest": "not-a-digest"},
    )


def experiment_module_qualname_only_identity_rejected() -> Invocation:
    return _audit_invocation("module_qualname_only_identity = True")


def experiment_changed_callable_source_changes_identity() -> Invocation:
    return _audit_invocation("filename_only_identity = True")


def experiment_request_reference_identity_mismatch() -> Invocation:
    context, backend, prepared = _prepared_execution()
    request = prepared._request
    assert request is not None
    reference = replace(request.callable_reference, callable_identity_digest="0" * 64)
    object.__setattr__(request, "callable_reference", reference)
    return Invocation(
        prepare_execution,
        (),
        {
            "context": context,
            "callable_binding": PORTABILITY_SCALE_ADD_BINDING,
            "request": request,
            "backend": backend,
        },
        PORTABILITY_SCALE_ADD_BINDING.reference.to_dict(),
        reference.to_dict(),
    )


def experiment_request_mode_options_mismatch() -> Invocation:
    context, backend, prepared = _prepared_execution()
    request = prepared._request
    assert request is not None
    object.__setattr__(request, "mode", "jit")
    return Invocation(
        prepare_execution,
        (),
        {
            "context": context,
            "callable_binding": PORTABILITY_SCALE_ADD_BINDING,
            "request": request,
            "backend": backend,
        },
        {"mode": "eager", "options_mode": "eager"},
        {"mode": "jit", "options_mode": "eager"},
    )


def experiment_input_signature_drift() -> Invocation:
    context, _, prepared = _prepared_execution()
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0, 2.0)},
        {"argument_count": 1},
        {"argument_count": 2},
    )


def experiment_compilation_options_drift() -> Invocation:
    options = CompilationOptions(mode="eager")
    context, prepared = _finalized_prepared(options=options)
    request = prepared._request
    assert request is not None
    object.__setattr__(request, "compilation_options", CompilationOptions(mode="jit"))
    object.__setattr__(request, "mode", "jit")
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0,)},
        {"mode": "eager"},
        {"mode": "jit"},
    )


def experiment_static_argument_value_drift() -> Invocation:
    options = CompilationOptions(mode="eager", static_arg_positions=(0,))
    context, prepared = _finalized_prepared(options=options)
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (3.0,)},
        {"static": 1.0},
        {"static": 3.0},
    )


def experiment_donation_contract_drift() -> Invocation:
    context, prepared = _finalized_prepared()
    request = prepared._request
    assert request is not None
    object.__setattr__(
        request,
        "compilation_options",
        CompilationOptions(mode="eager", donate_arg_positions=(0,)),
    )
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0,)},
        {"donation_positions": []},
        {"donation_positions": [0]},
    )


def experiment_placement_plan_drift() -> Invocation:
    context, prepared = _finalized_prepared()
    request = prepared._request
    assert request is not None
    object.__setattr__(request, "placement_plan_id", "other-placement")
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0,)},
        {"placement": "d-single-device"},
        {"placement": "other-placement"},
    )


def experiment_backend_identity_drift() -> Invocation:
    context, _, prepared = _prepared_execution()
    request = prepared._request
    assert request is not None
    foreign = _IdentityBackend("other-backend")
    return Invocation(
        prepare_execution,
        (),
        {
            "context": context,
            "callable_binding": PORTABILITY_SCALE_ADD_BINDING,
            "request": request,
            "backend": foreign,
        },
        {"backend": context.backend_id},
        {"backend": foreign.backend_id},
    )


def experiment_runtime_implementation_version_drift() -> Invocation:
    context, prepared = _finalized_prepared()
    prepared._backend.implementation_version = "2"
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0,)},
        {"runtime_implementation_version": "1"},
        {"runtime_implementation_version": "2"},
    )


def experiment_runtime_context_identity_drift() -> Invocation:
    _, prepared = _finalized_prepared()
    context = _execution_context(runtime_id="foreign-runtime")
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": prepared, "args": (1.0,)},
        {"runtime_id": "d-test-runtime"},
        {"runtime_id": "foreign-runtime"},
    )


def experiment_stale_prepared_execution_rejected() -> Invocation:
    context, prepared = _finalized_prepared()
    stale = replace(prepared)
    object.__setattr__(stale.prepared_identity, "prepared_execution_digest", "0" * 64)
    return Invocation(
        execute_prepared,
        (),
        {"context": context, "prepared": stale, "args": (1.0,)},
        {"prepared": "fresh"},
        {"prepared": "stale"},
    )


def experiment_cache_key_collision_rejected() -> Invocation:
    _, prepared = _finalized_prepared()
    cache = PreparedExecutionIdentityCache()
    cache.record(prepared.prepared_identity, cache_policy="reuse")
    conflicting = replace(prepared.prepared_identity)
    object.__setattr__(conflicting, "runtime_id", "foreign-runtime")
    return Invocation(
        cache.record,
        (conflicting,),
        {"cache_policy": "reuse"},
        {"cache": "fresh"},
        {"cache": "digest-collision"},
    )


def experiment_cache_disabled_reuse_claim_rejected() -> Invocation:
    _, prepared = _finalized_prepared()
    cache = PreparedExecutionIdentityCache()
    cache.record(prepared.prepared_identity, cache_policy="disabled")
    return Invocation(
        cache.record,
        (prepared.prepared_identity,),
        {"cache_policy": "disabled"},
        {"cache": "one-entry"},
        {"cache": "reuse-claimed-while-disabled"},
    )


def experiment_validation_observed_identity_injection() -> Invocation:
    return _audit_invocation("validation_observed_identity = value")


def experiment_permissive_callable_family_matcher() -> Invocation:
    return _audit_invocation("def _matches_expected(): pass")


def experiment_repr_or_object_id_identity_detected() -> Invocation:
    return _audit_invocation("value = repr(function)")


def experiment_production_imports_d_validation() -> Invocation:
    return _audit_invocation("from radjax_student.validation import x")


def experiment_competing_callable_identity_authority_detected() -> Invocation:
    return _audit_invocation("def bind_runtime_callable_two(): pass")


def _other_scale(value: float, increment: float) -> float:
    return value + increment
