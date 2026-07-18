"""Controlled preparation and execution of pure runtime functions."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from radjax_student.runtime.callables import (
    RuntimeCallableBinding,
    RuntimeCallableReference,
    RuntimePreparedExecutionIdentity,
    final_prepared_execution_identity,
)
from radjax_student.runtime.errors import RuntimeIssue
from radjax_student.runtime.models import (
    CompilationOptions,
    ExecutionContext,
    RuntimeCapabilityProfile,
    freeze_json_mapping,
    json_value,
)

ExecutionMode = Literal["eager", "jit", "automatic"]
ExecutionStatus = Literal["pass", "fail"]
EXECUTION_CAPABILITY_MAPPING_VERSION = "execution_capabilities.v1"
EXECUTION_CAPABILITY_MAPPING: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "eager": ("execution.eager_v1",),
        "jit": ("compilation.jit_v1",),
        "automatic": (),
    }
)
EXECUTION_BLOCKER_CODES: tuple[str, ...] = (
    "execution_mode_unsupported",
    "execution_request_invalid",
    "execution_function_missing",
    "execution_context_mismatch",
    "execution_capability_missing",
    "execution_static_argument_invalid",
    "execution_donation_invalid",
    "execution_preparation_failed",
    "execution_compilation_failed",
    "execution_dispatch_failed",
    "execution_synchronization_failed",
    "execution_output_invalid",
    "execution_internal_error",
    "execution_callable_declaration_invalid",
    "execution_callable_unsupported",
    "execution_callable_unregistered",
    "execution_callable_identity_mismatch",
    "execution_callable_source_unavailable",
    "execution_callable_source_invalid",
    "execution_callable_reference_invalid",
    "execution_callable_request_mismatch",
    "execution_prepared_identity_mismatch",
    "execution_backend_identity_mismatch",
    "execution_runtime_identity_mismatch",
    "execution_compilation_identity_mismatch",
    "execution_input_signature_mismatch",
    "execution_static_argument_identity_mismatch",
    "execution_donation_identity_mismatch",
    "execution_placement_identity_mismatch",
    "execution_cache_identity_mismatch",
    "execution_result_identity_mismatch",
    "execution_initialization_key_identity_mismatch",
)
EXECUTION_WARNING_CODES: tuple[str, ...] = (
    "execution_timings_not_benchmark",
    "execution_automatic_mode_unresolved",
    "execution_not_synchronized",
    "execution_donation_declared_not_proven",
    "execution_compilation_cache_unverified",
    "execution_declaration_not_training_proof",
)
EXECUTION_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "gradient_not_computed",
    "optimizer_not_updated",
    "training_not_run",
    "compiled_performance_not_benchmarked",
    "compilation_cache_not_persisted",
    "multi_device_execution_not_tested",
    "distributed_execution_not_tested",
    "model_function_not_proven",
)


class ExecutionBoundaryError(ValueError):
    """Structured preparation error with a stable execution blocker code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in EXECUTION_BLOCKER_CODES:
            raise ValueError(f"unknown execution blocker code: {code}")
        self.issue = RuntimeIssue.create(code, message, **details)
        super().__init__(f"{code}: {message}")

    @property
    def code(self) -> str:
        return self.issue.code

    def to_dict(self) -> dict[str, Any]:
        return self.issue.to_dict()


class ExecutionBackend(Protocol):
    backend_id: str

    def capability_profile(self) -> RuntimeCapabilityProfile: ...

    def prepare_runtime_execution(
        self,
        context: ExecutionContext,
        function: Callable[..., Any],
        request: ExecutionRequest,
        mode: ExecutionMode,
    ) -> Any: ...

    def compile_runtime_execution(
        self,
        context: ExecutionContext,
        handle: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[Any, bool]: ...

    def dispatch_runtime_execution(
        self,
        context: ExecutionContext,
        handle: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    def synchronize_runtime_execution(
        self,
        context: ExecutionContext,
        output: Any,
    ) -> Any: ...


class PreparedExecutionIdentityCache:
    """In-process exact-identity cache authority; it stores no executable handle."""

    def __init__(self) -> None:
        self._identities: dict[str, RuntimePreparedExecutionIdentity] = {}

    def record(
        self,
        identity: RuntimePreparedExecutionIdentity,
        *,
        cache_policy: str,
    ) -> bool:
        if not isinstance(identity, RuntimePreparedExecutionIdentity):
            raise ExecutionBoundaryError(
                "execution_cache_identity_mismatch",
                "cache identity must be RuntimePreparedExecutionIdentity",
            )
        if cache_policy not in {"reuse", "disabled"}:
            raise ExecutionBoundaryError(
                "execution_cache_identity_mismatch", "cache policy is invalid"
            )
        key = identity.prepared_execution_digest
        existing = self._identities.get(key)
        if existing is not None and existing != identity:
            raise ExecutionBoundaryError(
                "execution_cache_identity_mismatch",
                "prepared execution digest collides with different identity evidence",
            )
        if cache_policy == "disabled" and existing is not None:
            raise ExecutionBoundaryError(
                "execution_cache_identity_mismatch",
                "disabled cache policy cannot claim reuse",
            )
        if existing is None:
            self._identities[key] = identity
            return False
        return True


@dataclass(frozen=True)
class ExecutionRequest:
    """Serializable caller intent; the callable is deliberately separate."""

    request_id: str
    function_id: str
    mode: ExecutionMode
    compilation_options: CompilationOptions
    placement_plan_id: str | None = None
    input_signature: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    required_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    callable_reference: RuntimeCallableReference | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.request_id, "request_id")
        _require_identifier(self.function_id, "function_id")
        if self.callable_reference is not None:
            if not isinstance(self.callable_reference, RuntimeCallableReference):
                raise TypeError("callable_reference must be RuntimeCallableReference")
            if self.function_id != self.callable_reference.callable_id:
                raise ExecutionBoundaryError(
                    "execution_callable_request_mismatch",
                    "function_id must be derived from callable reference",
                )
        _require_mode(self.mode)
        if not isinstance(self.compilation_options, CompilationOptions):
            raise TypeError("compilation_options must be CompilationOptions")
        if self.mode != self.compilation_options.mode:
            raise ExecutionBoundaryError(
                "execution_compilation_identity_mismatch",
                "execution request mode must match compilation options mode",
                request_mode=self.mode,
                options_mode=self.compilation_options.mode,
            )
        if self.placement_plan_id is not None:
            _require_identifier(self.placement_plan_id, "placement_plan_id")
        if not isinstance(self.input_signature, Mapping):
            raise TypeError("input_signature must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("execution request metadata must be a mapping")
        capabilities = _sorted_strings(
            (
                *self.required_capabilities,
                *execution_capabilities(self.mode, self.compilation_options),
            ),
            "required_capabilities",
        )
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(
            self, "input_signature", freeze_json_mapping(self.input_signature)
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "function_id": self.function_id,
            "mode": self.mode,
            "compilation_options": self.compilation_options.to_dict(),
            "placement_plan_id": self.placement_plan_id,
            "input_signature": json_value(self.input_signature),
            "required_capabilities": list(self.required_capabilities),
            "metadata": json_value(self.metadata),
            "callable_reference": None
            if self.callable_reference is None
            else self.callable_reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionRequest:
        required = {
            "request_id",
            "function_id",
            "mode",
            "compilation_options",
            "placement_plan_id",
            "input_signature",
            "required_capabilities",
            "metadata",
            "callable_reference",
        }
        if set(payload) != required:
            raise ValueError("execution request fields are missing or unknown")
        return cls(
            request_id=_string(payload["request_id"], "request_id"),
            function_id=_string(payload["function_id"], "function_id"),
            mode=_string(payload["mode"], "mode"),
            compilation_options=CompilationOptions.from_dict(
                _mapping(payload["compilation_options"], "compilation_options")
            ),
            placement_plan_id=_optional_string(payload.get("placement_plan_id")),
            input_signature=_mapping(
                payload.get("input_signature", {}), "input_signature"
            ),
            required_capabilities=_strings(
                payload.get("required_capabilities", ()), "required_capabilities"
            ),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
            callable_reference=(
                None
                if payload.get("callable_reference") is None
                else RuntimeCallableReference.from_dict(
                    _mapping(payload["callable_reference"], "callable_reference")
                )
            ),
        )


@dataclass(frozen=True)
class PreparedExecution:
    """Opaque runtime-owned preparation with serializable metadata only."""

    backend_id: str
    function_id: str
    mode: ExecutionMode
    compiled: bool
    capabilities: tuple[str, ...]
    preparation_metadata: Mapping[str, Any]
    preparation_seconds: float
    warnings: tuple[RuntimeIssue, ...] = ()
    callable_reference: RuntimeCallableReference | None = None
    prepared_identity: RuntimePreparedExecutionIdentity | None = None
    _handle: Any = field(repr=False, compare=False, default=None)
    _backend: Any = field(repr=False, compare=False, default=None)
    _request: ExecutionRequest | None = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        _require_identifier(self.backend_id, "backend_id")
        _require_identifier(self.function_id, "function_id")
        _require_mode(self.mode)
        if not isinstance(self.compiled, bool):
            raise TypeError("compiled must be boolean")
        capabilities = _sorted_strings(self.capabilities, "capabilities")
        if not isinstance(self.preparation_metadata, Mapping):
            raise TypeError("preparation_metadata must be a mapping")
        _require_duration(self.preparation_seconds, "preparation_seconds")
        warnings = _issues(self.warnings, "warnings", EXECUTION_WARNING_CODES)
        if self.callable_reference is not None and not isinstance(
            self.callable_reference, RuntimeCallableReference
        ):
            raise TypeError("callable_reference must be RuntimeCallableReference")
        if self.prepared_identity is not None and not isinstance(
            self.prepared_identity, RuntimePreparedExecutionIdentity
        ):
            raise TypeError(
                "prepared_identity must be RuntimePreparedExecutionIdentity"
            )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "preparation_metadata",
            freeze_json_mapping(self.preparation_metadata),
        )
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "function_id": self.function_id,
            "mode": self.mode,
            "compiled": self.compiled,
            "capabilities": list(self.capabilities),
            "preparation_metadata": json_value(self.preparation_metadata),
            "preparation_seconds": self.preparation_seconds,
            "warnings": [item.to_dict() for item in self.warnings],
            "callable_reference": None
            if self.callable_reference is None
            else self.callable_reference.to_dict(),
            "prepared_identity": None
            if self.prepared_identity is None
            else self.prepared_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreparedExecution:
        required = {
            "backend_id",
            "function_id",
            "mode",
            "compiled",
            "capabilities",
            "preparation_metadata",
            "preparation_seconds",
            "warnings",
            "callable_reference",
            "prepared_identity",
        }
        if set(payload) != required:
            raise ValueError("prepared execution fields are missing or unknown")
        return cls(
            backend_id=_string(payload["backend_id"], "backend_id"),
            function_id=_string(payload["function_id"], "function_id"),
            mode=_string(payload["mode"], "mode"),
            compiled=_bool(payload["compiled"], "compiled"),
            capabilities=_strings(payload.get("capabilities", ()), "capabilities"),
            preparation_metadata=_mapping(
                payload.get("preparation_metadata", {}), "preparation_metadata"
            ),
            preparation_seconds=_number(
                payload["preparation_seconds"], "preparation_seconds"
            ),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            callable_reference=(
                None
                if payload.get("callable_reference") is None
                else RuntimeCallableReference.from_dict(
                    _mapping(payload["callable_reference"], "callable_reference")
                )
            ),
            prepared_identity=(
                None
                if payload.get("prepared_identity") is None
                else RuntimePreparedExecutionIdentity.from_dict(
                    _mapping(payload["prepared_identity"], "prepared_identity")
                )
            ),
        )


@dataclass(frozen=True)
class ExecutionResult:
    """Structured execution report with no callable, handle, or raw output."""

    status: ExecutionStatus
    request_id: str
    backend_id: str
    mode: ExecutionMode
    compiled: bool
    dispatched: bool
    synchronized: bool
    output_metadata: Mapping[str, Any]
    preparation_seconds: float
    compilation_seconds: float
    dispatch_seconds: float
    synchronization_seconds: float
    total_seconds: float
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = EXECUTION_CLAIMS_NOT_MADE
    callable_reference: RuntimeCallableReference | None = None
    prepared_execution_digest: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("execution status must be pass or fail")
        _require_identifier(self.request_id, "request_id")
        _require_identifier(self.backend_id, "backend_id")
        _require_mode(self.mode)
        for name in ("compiled", "dispatched", "synchronized"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.output_metadata, Mapping):
            raise TypeError("output_metadata must be a mapping")
        for name in (
            "preparation_seconds",
            "compilation_seconds",
            "dispatch_seconds",
            "synchronization_seconds",
            "total_seconds",
        ):
            _require_duration(getattr(self, name), name)
        blockers = _issues(self.blockers, "blockers", EXECUTION_BLOCKER_CODES)
        warnings = _issues(self.warnings, "warnings", EXECUTION_WARNING_CODES)
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if self.status == "pass" and blockers:
            raise ValueError("passing execution cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing execution must contain blockers")
        if self.callable_reference is not None and not isinstance(
            self.callable_reference, RuntimeCallableReference
        ):
            raise TypeError("callable_reference must be RuntimeCallableReference")
        if self.prepared_execution_digest is not None:
            _require_digest(self.prepared_execution_digest, "prepared_execution_digest")
        if (
            self.status == "pass"
            and self.callable_reference is not None
            and self.prepared_execution_digest is None
        ):
            raise ValueError(
                "passing execution requires final prepared execution identity"
            )
        object.__setattr__(
            self, "output_metadata", freeze_json_mapping(self.output_metadata)
        )
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "backend_id": self.backend_id,
            "mode": self.mode,
            "compiled": self.compiled,
            "dispatched": self.dispatched,
            "synchronized": self.synchronized,
            "output_metadata": json_value(self.output_metadata),
            "preparation_seconds": self.preparation_seconds,
            "compilation_seconds": self.compilation_seconds,
            "dispatch_seconds": self.dispatch_seconds,
            "synchronization_seconds": self.synchronization_seconds,
            "total_seconds": self.total_seconds,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
            "callable_reference": None
            if self.callable_reference is None
            else self.callable_reference.to_dict(),
            "prepared_execution_digest": self.prepared_execution_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionResult:
        required = {
            "status",
            "request_id",
            "backend_id",
            "mode",
            "compiled",
            "dispatched",
            "synchronized",
            "output_metadata",
            "preparation_seconds",
            "compilation_seconds",
            "dispatch_seconds",
            "synchronization_seconds",
            "total_seconds",
            "blockers",
            "warnings",
            "claims_not_made",
            "callable_reference",
            "prepared_execution_digest",
        }
        if set(payload) != required:
            raise ValueError("execution result fields are missing or unknown")
        return cls(
            status=_string(payload["status"], "status"),
            request_id=_string(payload["request_id"], "request_id"),
            backend_id=_string(payload["backend_id"], "backend_id"),
            mode=_string(payload["mode"], "mode"),
            compiled=_bool(payload["compiled"], "compiled"),
            dispatched=_bool(payload["dispatched"], "dispatched"),
            synchronized=_bool(payload["synchronized"], "synchronized"),
            output_metadata=_mapping(payload["output_metadata"], "output_metadata"),
            preparation_seconds=_number(
                payload["preparation_seconds"], "preparation_seconds"
            ),
            compilation_seconds=_number(
                payload["compilation_seconds"], "compilation_seconds"
            ),
            dispatch_seconds=_number(payload["dispatch_seconds"], "dispatch_seconds"),
            synchronization_seconds=_number(
                payload["synchronization_seconds"], "synchronization_seconds"
            ),
            total_seconds=_number(payload["total_seconds"], "total_seconds"),
            blockers=_issues_from_payload(payload.get("blockers", ()), "blockers"),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()), "claims_not_made"
            ),
            callable_reference=(
                None
                if payload.get("callable_reference") is None
                else RuntimeCallableReference.from_dict(
                    _mapping(payload["callable_reference"], "callable_reference")
                )
            ),
            prepared_execution_digest=_optional_string(
                payload.get("prepared_execution_digest")
            ),
        )


def execution_capabilities(
    mode: ExecutionMode,
    options: CompilationOptions,
) -> tuple[str, ...]:
    """Return centralized versioned capability requirements for one request."""

    _require_mode(mode)
    if not isinstance(options, CompilationOptions):
        raise TypeError("options must be CompilationOptions")
    required = list(EXECUTION_CAPABILITY_MAPPING[mode])
    if options.static_arg_names or options.static_arg_positions:
        required.append("execution.static_arguments_v1")
    if options.donate_arg_names or options.donate_arg_positions:
        required.append("execution.argument_donation_v1")
    if options.synchronize_results:
        required.append("execution.synchronize_v1")
    return tuple(sorted(set(required)))


def prepare_execution(
    *,
    context: ExecutionContext,
    function: Callable[..., Any] | None = None,
    callable_binding: RuntimeCallableBinding | None = None,
    request: ExecutionRequest,
    backend: ExecutionBackend,
) -> PreparedExecution:
    """Validate policy and create an opaque backend preparation without executing."""

    if callable_binding is not None:
        function = callable_binding.callable
        if request.callable_reference != callable_binding.reference:
            raise ExecutionBoundaryError(
                "execution_callable_request_mismatch",
                "request callable reference does not match binding",
            )
    elif request.callable_reference is not None:
        raise ExecutionBoundaryError(
            "execution_callable_request_mismatch",
            "a callable reference requires a runtime callable binding",
        )
    if function is None:
        raise ExecutionBoundaryError(
            "execution_function_missing", "execution function must be callable"
        )
    _validate_preparation_inputs(context, function, request, backend)
    mode, warnings = _effective_mode(request)
    _validate_signature(function, request.compilation_options)
    _validate_capabilities(
        backend.capability_profile(),
        tuple(
            sorted(
                set(
                    (
                        *request.required_capabilities,
                        *EXECUTION_CAPABILITY_MAPPING[mode],
                    )
                )
            )
        ),
    )
    start = time.perf_counter()
    try:
        handle = backend.prepare_runtime_execution(context, function, request, mode)
    except Exception as exc:
        raise _execution_error(
            "execution_preparation_failed",
            "runtime backend could not prepare the requested execution",
            exc,
        ) from exc
    preparation_seconds = time.perf_counter() - start
    return PreparedExecution(
        backend_id=backend.backend_id,
        function_id=request.function_id,
        mode=mode,
        compiled=False,
        capabilities=request.required_capabilities,
        preparation_metadata={
            "cache_policy": request.compilation_options.cache_policy,
            "requested_mode": request.mode,
            "placement_plan_id": request.placement_plan_id,
        },
        preparation_seconds=preparation_seconds,
        warnings=_base_warnings(request, warnings),
        _handle=handle,
        _backend=backend,
        _request=request,
        callable_reference=request.callable_reference,
    )


def execute_prepared(
    *,
    context: ExecutionContext,
    prepared: PreparedExecution,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any | None, ExecutionResult]:
    """Compile if needed, dispatch, optionally synchronize, and report each phase."""

    arguments = tuple(args)
    keyword_arguments = {} if kwargs is None else dict(kwargs)
    request = prepared._request
    backend = prepared._backend
    if request is None or backend is None or context.backend_id != prepared.backend_id:
        return None, _failure_result(
            request_id=request.request_id if request is not None else "unknown-request",
            backend_id=prepared.backend_id,
            mode=prepared.mode,
            preparation_seconds=prepared.preparation_seconds,
            warnings=prepared.warnings,
            blocker=RuntimeIssue.create(
                (
                    "execution_runtime_identity_mismatch"
                    if prepared.callable_reference is not None
                    else "execution_context_mismatch"
                ),
                "prepared execution does not belong to the supplied runtime context",
            ),
        )
    start_total = time.perf_counter()
    handle = prepared._handle
    compiled = prepared.compiled
    compilation_seconds = 0.0
    dispatch_seconds = 0.0
    synchronization_seconds = 0.0
    output: Any = None
    dispatched = False
    synchronized = False
    try:
        prepared_identity = None
        if request.callable_reference is not None:
            try:
                prepared_identity = _finalize_prepared_identity(
                    context, prepared, arguments, keyword_arguments
                )
            except ExecutionBoundaryError as exc:
                return None, _failure_result(
                    request_id=request.request_id,
                    backend_id=prepared.backend_id,
                    mode=prepared.mode,
                    preparation_seconds=prepared.preparation_seconds,
                    warnings=prepared.warnings,
                    blocker=exc.issue,
                    callable_reference=prepared.callable_reference,
                )
            if (
                prepared.prepared_identity is not None
                and prepared.prepared_identity != prepared_identity
            ):
                return None, _failure_result(
                    request_id=request.request_id,
                    backend_id=prepared.backend_id,
                    mode=prepared.mode,
                    preparation_seconds=prepared.preparation_seconds,
                    warnings=prepared.warnings,
                    blocker=RuntimeIssue.create(
                        _prepared_identity_mismatch_code(
                            prepared.prepared_identity, prepared_identity
                        ),
                        "prepared execution identity no longer matches invocation",
                    ),
                    callable_reference=prepared.callable_reference,
                )
            prepared = PreparedExecution(
                **{**prepared.__dict__, "prepared_identity": prepared_identity}
            )
        if prepared.mode == "jit":
            start = time.perf_counter()
            try:
                handle, _compiled_now = backend.compile_runtime_execution(
                    context,
                    handle,
                    arguments,
                    keyword_arguments,
                )
                compiled = True
            except Exception as exc:
                return None, _failure_result(
                    request_id=request.request_id,
                    backend_id=prepared.backend_id,
                    mode=prepared.mode,
                    preparation_seconds=prepared.preparation_seconds,
                    compilation_seconds=time.perf_counter() - start,
                    warnings=prepared.warnings,
                    blocker=_issue_from_exception(
                        "execution_compilation_failed",
                        "runtime backend could not compile the requested execution",
                        exc,
                    ),
                    callable_reference=prepared.callable_reference,
                    prepared_execution_digest=(
                        None
                        if prepared_identity is None
                        else prepared_identity.prepared_execution_digest
                    ),
                )
            compilation_seconds = time.perf_counter() - start

        start = time.perf_counter()
        try:
            output = backend.dispatch_runtime_execution(
                context,
                handle,
                arguments,
                keyword_arguments,
            )
            dispatched = True
        except Exception as exc:
            return None, _failure_result(
                request_id=request.request_id,
                backend_id=prepared.backend_id,
                mode=prepared.mode,
                preparation_seconds=prepared.preparation_seconds,
                compilation_seconds=compilation_seconds,
                dispatch_seconds=time.perf_counter() - start,
                warnings=prepared.warnings,
                blocker=_issue_from_exception(
                    "execution_dispatch_failed",
                    "runtime backend could not dispatch the prepared execution",
                    exc,
                ),
                callable_reference=prepared.callable_reference,
                prepared_execution_digest=(
                    None
                    if prepared_identity is None
                    else prepared_identity.prepared_execution_digest
                ),
            )
        dispatch_seconds = time.perf_counter() - start

        if request.compilation_options.synchronize_results:
            start = time.perf_counter()
            try:
                output = backend.synchronize_runtime_execution(context, output)
                synchronized = True
            except Exception as exc:
                return None, _failure_result(
                    request_id=request.request_id,
                    backend_id=prepared.backend_id,
                    mode=prepared.mode,
                    preparation_seconds=prepared.preparation_seconds,
                    compilation_seconds=compilation_seconds,
                    dispatch_seconds=dispatch_seconds,
                    synchronization_seconds=time.perf_counter() - start,
                    warnings=prepared.warnings,
                    blocker=_issue_from_exception(
                        "execution_synchronization_failed",
                        "runtime backend could not synchronize execution output",
                        exc,
                    ),
                    callable_reference=prepared.callable_reference,
                    prepared_execution_digest=(
                        None
                        if prepared_identity is None
                        else prepared_identity.prepared_execution_digest
                    ),
                )
            synchronization_seconds = time.perf_counter() - start

        try:
            output_metadata = output_metadata_for(output)
        except Exception as exc:
            return None, _failure_result(
                request_id=request.request_id,
                backend_id=prepared.backend_id,
                mode=prepared.mode,
                preparation_seconds=prepared.preparation_seconds,
                compilation_seconds=compilation_seconds,
                dispatch_seconds=dispatch_seconds,
                synchronization_seconds=synchronization_seconds,
                warnings=prepared.warnings,
                blocker=_issue_from_exception(
                    "execution_output_invalid",
                    "execution output could not be represented as metadata",
                    exc,
                ),
                callable_reference=prepared.callable_reference,
                prepared_execution_digest=(
                    None
                    if prepared_identity is None
                    else prepared_identity.prepared_execution_digest
                ),
            )
        return output, ExecutionResult(
            status="pass",
            request_id=request.request_id,
            backend_id=prepared.backend_id,
            mode=prepared.mode,
            compiled=compiled,
            dispatched=dispatched,
            synchronized=synchronized,
            output_metadata=output_metadata,
            preparation_seconds=prepared.preparation_seconds,
            compilation_seconds=compilation_seconds,
            dispatch_seconds=dispatch_seconds,
            synchronization_seconds=synchronization_seconds,
            total_seconds=prepared.preparation_seconds
            + (time.perf_counter() - start_total),
            warnings=_result_warnings(prepared.warnings, synchronized),
            callable_reference=prepared.callable_reference,
            prepared_execution_digest=(
                None
                if prepared_identity is None
                else prepared_identity.prepared_execution_digest
            ),
        )
    except Exception as exc:
        return None, _failure_result(
            request_id=request.request_id,
            backend_id=prepared.backend_id,
            mode=prepared.mode,
            preparation_seconds=prepared.preparation_seconds,
            compilation_seconds=compilation_seconds,
            dispatch_seconds=dispatch_seconds,
            synchronization_seconds=synchronization_seconds,
            warnings=prepared.warnings,
            blocker=_issue_from_exception(
                "execution_internal_error",
                "execution boundary failed outside a recognized phase",
                exc,
            ),
        )


def execute_function(
    *,
    context: ExecutionContext,
    function: Callable[..., Any] | None = None,
    callable_binding: RuntimeCallableBinding | None = None,
    request: ExecutionRequest,
    backend: ExecutionBackend,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any | None, ExecutionResult]:
    """Convenience path that turns preparation failure into a structured report."""

    try:
        prepared = prepare_execution(
            context=context,
            function=function,
            callable_binding=callable_binding,
            request=request,
            backend=backend,
        )
    except ExecutionBoundaryError as exc:
        return None, _failure_result(
            request_id=request.request_id,
            backend_id=getattr(backend, "backend_id", context.backend_id),
            mode=request.mode,
            blocker=exc.issue,
        )
    return execute_prepared(
        context=context,
        prepared=prepared,
        args=args,
        kwargs=kwargs,
    )


def output_metadata_for(output: Any) -> dict[str, Any]:
    """Summarize an output tree without retaining its values or backend object."""

    if hasattr(output, "shape") or hasattr(output, "dtype"):
        shape = getattr(output, "shape", None)
        return {
            "tree_structure_summary": "leaf",
            "shape": None if shape is None else tuple(int(item) for item in shape),
            "dtype": None
            if getattr(output, "dtype", None) is None
            else str(output.dtype),
            "device_ids": (),
            "sharding_summary": "unknown",
        }
    if isinstance(output, Mapping):
        return {
            "tree_structure_summary": "mapping",
            "leaf_count": len(output),
            "shape": None,
            "dtype": None,
            "device_ids": (),
            "sharding_summary": "unknown",
        }
    if isinstance(output, (list, tuple)):
        return {
            "tree_structure_summary": "sequence",
            "leaf_count": len(output),
            "shape": (len(output),),
            "dtype": None,
            "device_ids": (),
            "sharding_summary": "unknown",
        }
    if output is None or isinstance(output, (str, int, float, bool)):
        return {
            "tree_structure_summary": "scalar",
            "shape": (),
            "dtype": type(output).__name__,
            "device_ids": (),
            "sharding_summary": "unknown",
        }
    raise TypeError(
        f"unsupported execution output metadata type: {type(output).__name__}"
    )


def _validate_preparation_inputs(
    context: ExecutionContext,
    function: Callable[..., Any],
    request: ExecutionRequest,
    backend: ExecutionBackend,
) -> None:
    if not isinstance(context, ExecutionContext):
        raise ExecutionBoundaryError(
            "execution_context_mismatch", "execution requires an ExecutionContext"
        )
    if not callable(function):
        raise ExecutionBoundaryError(
            "execution_function_missing", "execution function must be callable"
        )
    if not isinstance(request, ExecutionRequest):
        raise ExecutionBoundaryError(
            "execution_request_invalid", "execution request must be ExecutionRequest"
        )
    if request.mode != request.compilation_options.mode:
        raise ExecutionBoundaryError(
            "execution_compilation_identity_mismatch",
            "execution request mode does not match compilation options",
        )
    if getattr(backend, "backend_id", None) != context.backend_id:
        raise ExecutionBoundaryError(
            (
                "execution_backend_identity_mismatch"
                if request.callable_reference is not None
                else "execution_context_mismatch"
            ),
            "execution backend does not match runtime context backend",
            context_backend_id=context.backend_id,
            backend_id=getattr(backend, "backend_id", None),
        )


def _finalize_prepared_identity(
    context: ExecutionContext,
    prepared: PreparedExecution,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> RuntimePreparedExecutionIdentity:
    request = prepared._request
    if request is None or request.callable_reference is None:
        raise ExecutionBoundaryError(
            "execution_callable_request_mismatch",
            "runtime callable reference is required before dispatch",
        )
    options = request.compilation_options
    _validate_input_signature(request.input_signature, args, kwargs)
    static_contract = {
        "names": list(options.static_arg_names),
        "positions": list(options.static_arg_positions),
    }
    static_values = _static_argument_values(args, kwargs, options)
    implicit_positions = [
        position
        for position, value in enumerate(args)
        if _is_jax_callable_partial(value)
    ]
    if implicit_positions:
        static_contract["implicit_jax_partial_positions"] = implicit_positions
        static_values.update(
            {
                f"implicit_jax_partial:{position}": _canonical_static_value(
                    args[position]
                )
                for position in implicit_positions
            }
        )
    donation = {
        "names": list(options.donate_arg_names),
        "positions": list(options.donate_arg_positions),
    }
    version = getattr(prepared._backend, "implementation_version", None)
    if not isinstance(version, str) or not version:
        version = str(context.metadata.get("runtime_implementation_version", "unknown"))
    return final_prepared_execution_identity(
        reference=request.callable_reference,
        backend_id=prepared.backend_id,
        runtime_id=context.runtime_id,
        runtime_implementation_version=version,
        mode=prepared.mode,
        compilation_options=options.to_dict(),
        input_signature=dict(request.input_signature),
        static_contract=static_contract,
        static_values=static_values,
        donation_contract=donation,
        placement_plan_identity=request.placement_plan_id,
        required_capabilities=request.required_capabilities,
    )


def finalize_prepared_execution_identity(
    *,
    context: ExecutionContext,
    prepared: PreparedExecution,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> RuntimePreparedExecutionIdentity:
    """Public runtime finalization authority for cache identity inspection.

    The result is derived only when the actual invocation arguments exist; it
    never creates a placeholder identity during preparation.
    """
    return _finalize_prepared_identity(
        context,
        prepared,
        tuple(args),
        {} if kwargs is None else dict(kwargs),
    )


def _validate_input_signature(
    input_signature: Mapping[str, Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    """Enforce optional structural invocation facts without hashing array values."""
    expected_count = input_signature.get("argument_count")
    if expected_count is not None and expected_count != len(args):
        raise ExecutionBoundaryError(
            "execution_input_signature_mismatch",
            "runtime invocation argument count differs from input contract",
        )
    expected_keywords = input_signature.get("keyword_names")
    if expected_keywords is not None and tuple(expected_keywords) != tuple(
        sorted(kwargs)
    ):
        raise ExecutionBoundaryError(
            "execution_input_signature_mismatch",
            "runtime invocation keyword names differ from input contract",
        )


def _prepared_identity_mismatch_code(
    previous: RuntimePreparedExecutionIdentity,
    current: RuntimePreparedExecutionIdentity,
) -> str:
    if previous.backend_id != current.backend_id:
        return "execution_backend_identity_mismatch"
    if (
        previous.runtime_id != current.runtime_id
        or previous.runtime_implementation_version
        != current.runtime_implementation_version
    ):
        return "execution_runtime_identity_mismatch"
    if (
        previous.static_argument_contract_digest
        != current.static_argument_contract_digest
    ):
        return "execution_static_argument_identity_mismatch"
    if previous.static_argument_value_digest != current.static_argument_value_digest:
        return "execution_static_argument_identity_mismatch"
    if previous.donation_contract_digest != current.donation_contract_digest:
        return "execution_donation_identity_mismatch"
    if previous.placement_plan_identity != current.placement_plan_identity:
        return "execution_placement_identity_mismatch"
    if previous.input_signature_digest != current.input_signature_digest:
        return "execution_input_signature_mismatch"
    if previous.compilation_options_digest != current.compilation_options_digest:
        return "execution_compilation_identity_mismatch"
    return "execution_prepared_identity_mismatch"


def _static_argument_values(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], options: CompilationOptions
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for position in options.static_arg_positions:
        if position >= len(args):
            raise ExecutionBoundaryError(
                "execution_static_argument_identity_mismatch",
                "declared static position is absent from invocation",
            )
        values[f"position:{position}"] = _canonical_static_value(args[position])
    for name in options.static_arg_names:
        if name not in kwargs:
            raise ExecutionBoundaryError(
                "execution_static_argument_identity_mismatch",
                "declared static name is absent from invocation",
            )
        values[f"name:{name}"] = _canonical_static_value(kwargs[name])
    return values


def _canonical_static_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, tuple):
        return ["tuple", [_canonical_static_value(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_canonical_static_value(item) for item in value]]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _canonical_static_value(value[key]) for key in sorted(value)}
    if _is_jax_callable_partial(value):
        function = value.func
        if not inspect.isfunction(function):
            raise ExecutionBoundaryError(
                "execution_static_argument_identity_mismatch",
                "JAX partial must wrap a Python function",
            )
        try:
            source = inspect.getsource(function)
            tree = ast.parse(inspect.cleandoc(source))
        except (OSError, TypeError, SyntaxError) as exc:
            raise ExecutionBoundaryError(
                "execution_static_argument_identity_mismatch",
                "JAX partial source is unavailable",
            ) from exc
        return {
            "kind": "jax_tree_partial",
            "function_module": function.__module__,
            "function_qualname": function.__qualname__,
            "function_source_digest": hashlib.sha256(
                ast.dump(tree, annotate_fields=True, include_attributes=False).encode()
            ).hexdigest(),
            "closure_values": _closure_static_values(function),
        }
    raise ExecutionBoundaryError(
        "execution_static_argument_identity_mismatch",
        "static argument has no canonical identity",
    )


def _is_jax_callable_partial(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "jax.tree_util" and value_type.__name__ == "Partial"


def _closure_static_values(function: Callable[..., Any]) -> dict[str, Any]:
    cells = function.__closure__ or ()
    names = function.__code__.co_freevars
    return {
        name: _closure_static_value(cell.cell_contents)
        for name, cell in zip(names, cells, strict=True)
    }


def _closure_static_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, tuple, Mapping)):
        return _canonical_static_value(value)
    if inspect.isfunction(value):
        try:
            source = ast.parse(inspect.cleandoc(inspect.getsource(value)))
        except (OSError, TypeError, SyntaxError) as exc:
            raise ExecutionBoundaryError(
                "execution_static_argument_identity_mismatch",
                "nested static callable source is unavailable",
            ) from exc
        return {
            "kind": "function",
            "module": value.__module__,
            "qualname": value.__qualname__,
            "source_digest": hashlib.sha256(
                ast.dump(
                    source, annotate_fields=True, include_attributes=False
                ).encode()
            ).hexdigest(),
            "closure_values": _closure_static_values(value),
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {"kind": "typed", "value": _canonical_static_value(value.to_dict())}
    if (
        type(value).__name__ == "JaxExecutionPlan"
        and hasattr(value, "objective_selection")
        and hasattr(value, "update_selection")
        and hasattr(value, "parameter_layout")
    ):
        return {
            "kind": "jax_execution_plan",
            "objective_selection": value.objective_selection.digest,
            "update_selection": _canonical_static_value(
                value.update_selection.to_dict()
            ),
            "parameter_layout": value.parameter_layout.digest(),
        }
    identity = {
        name: getattr(value, name)
        for name in (
            "architecture_id",
            "architecture_version",
            "objective_id",
            "objective_version",
            "optimizer_id",
            "optimizer_version",
        )
        if hasattr(value, name) and isinstance(getattr(value, name), (str, int))
    }
    if identity:
        return {"kind": "owner_identity", "value": identity}
    raise ExecutionBoundaryError(
        "execution_static_argument_identity_mismatch",
        "static closure value has no canonical identity: "
        f"{type(value).__module__}.{type(value).__name__}",
    )


def _effective_mode(
    request: ExecutionRequest,
) -> tuple[ExecutionMode, tuple[RuntimeIssue, ...]]:
    if request.mode == "automatic":
        return (
            "eager",
            (
                RuntimeIssue.create(
                    "execution_automatic_mode_unresolved",
                    "automatic execution mode resolves to eager in P2.7; "
                    "JIT remains explicit",
                ),
            ),
        )
    return request.mode, ()


def _validate_signature(
    function: Callable[..., Any],
    options: CompilationOptions,
) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    parameters = tuple(signature.parameters.values())
    parameter_names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    accepts_args = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    for name in (*options.static_arg_names, *options.donate_arg_names):
        if name not in parameter_names and not accepts_kwargs:
            raise ExecutionBoundaryError(
                "execution_static_argument_invalid"
                if name in options.static_arg_names
                else "execution_donation_invalid",
                "declared argument name is not present in the function signature",
                argument_name=name,
            )
    if not accepts_args:
        maximum_position = len(parameters) - 1
        for position in (*options.static_arg_positions, *options.donate_arg_positions):
            if position > maximum_position:
                raise ExecutionBoundaryError(
                    "execution_static_argument_invalid"
                    if position in options.static_arg_positions
                    else "execution_donation_invalid",
                    "declared argument position is outside the function signature",
                    argument_position=position,
                )


def _validate_capabilities(
    profile: RuntimeCapabilityProfile,
    required: tuple[str, ...],
) -> None:
    missing = tuple(item for item in required if item not in profile.capabilities)
    if missing:
        raise ExecutionBoundaryError(
            "execution_capability_missing",
            "runtime backend does not declare all required execution capabilities",
            backend_id=profile.backend_id,
            missing_capabilities=missing,
        )


def _base_warnings(
    request: ExecutionRequest,
    mode_warnings: tuple[RuntimeIssue, ...],
) -> tuple[RuntimeIssue, ...]:
    warnings = [
        RuntimeIssue.create(
            "execution_timings_not_benchmark",
            "execution timing fields are diagnostic observations, not benchmarks",
        ),
        RuntimeIssue.create(
            "execution_declaration_not_training_proof",
            "pure function execution does not prove model or training behavior",
        ),
        *mode_warnings,
    ]
    options = request.compilation_options
    if options.donate_arg_names or options.donate_arg_positions:
        warnings.append(
            RuntimeIssue.create(
                "execution_donation_declared_not_proven",
                "argument donation is declared but memory effects are not proven",
            )
        )
    if options.cache_policy == "reuse":
        warnings.append(
            RuntimeIssue.create(
                "execution_compilation_cache_unverified",
                "compilation cache reuse is requested but not externally verified",
            )
        )
    return _deduplicate_issues(warnings)


def _result_warnings(
    warnings: tuple[RuntimeIssue, ...],
    synchronized: bool,
) -> tuple[RuntimeIssue, ...]:
    result = list(warnings)
    if not synchronized:
        result.append(
            RuntimeIssue.create(
                "execution_not_synchronized",
                "execution output was dispatched without an explicit completion wait",
            )
        )
    return _deduplicate_issues(result)


def _failure_result(
    *,
    request_id: str,
    backend_id: str,
    mode: ExecutionMode,
    preparation_seconds: float = 0.0,
    compilation_seconds: float = 0.0,
    dispatch_seconds: float = 0.0,
    synchronization_seconds: float = 0.0,
    warnings: tuple[RuntimeIssue, ...] = (),
    blocker: RuntimeIssue,
    callable_reference: RuntimeCallableReference | None = None,
    prepared_execution_digest: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        status="fail",
        request_id=request_id,
        backend_id=backend_id,
        mode=mode,
        compiled=False,
        dispatched=False,
        synchronized=False,
        output_metadata={},
        preparation_seconds=preparation_seconds,
        compilation_seconds=compilation_seconds,
        dispatch_seconds=dispatch_seconds,
        synchronization_seconds=synchronization_seconds,
        total_seconds=(
            preparation_seconds
            + compilation_seconds
            + dispatch_seconds
            + synchronization_seconds
        ),
        blockers=(blocker,),
        warnings=_result_warnings(warnings, False),
        callable_reference=callable_reference,
        prepared_execution_digest=prepared_execution_digest,
    )


def _execution_error(code: str, message: str, exc: Exception) -> ExecutionBoundaryError:
    if isinstance(exc, ExecutionBoundaryError):
        return exc
    return ExecutionBoundaryError(
        code,
        message,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )


def _issue_from_exception(code: str, message: str, exc: Exception) -> RuntimeIssue:
    if isinstance(exc, ExecutionBoundaryError):
        return exc.issue
    return RuntimeIssue.create(
        code,
        message,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )


def _require_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_mode(mode: object) -> None:
    if mode not in ("eager", "jit", "automatic"):
        raise ExecutionBoundaryError(
            "execution_mode_unsupported",
            "execution mode must be eager, jit, or automatic",
            mode=mode if isinstance(mode, str) else None,
        )


def _require_duration(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value, "value")


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _number(value: object, name: str) -> float:
    _require_duration(value, name)
    return float(value)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _unique_strings(value: object, name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _sorted_strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(sorted(set(_strings(value, name))))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _issues(
    value: object,
    name: str,
    valid_codes: tuple[str, ...],
) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    invalid = [item.code for item in result if item.code not in valid_codes]
    if invalid:
        raise ValueError(f"{name} contains unknown execution codes: {invalid}")
    return result


def _issues_from_payload(value: object, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(RuntimeIssue.from_dict(_mapping(item, name)) for item in value)


def _deduplicate_issues(issues: list[RuntimeIssue]) -> tuple[RuntimeIssue, ...]:
    result: list[RuntimeIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        identity = (issue.code, issue.message, repr(sorted(issue.details.items())))
        if identity not in seen:
            result.append(issue)
            seen.add(identity)
    return tuple(result)
