"""Controlled preparation and execution of pure runtime functions."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

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


@dataclass(frozen=True)
class ExecutionRequest:
    """Serializable caller intent; the callable is deliberately separate."""

    request_id: str
    function_id: str
    mode: ExecutionMode
    compilation_options: CompilationOptions
    placement_plan_id: str | None = None
    input_signature: Mapping[str, Any] = MappingProxyType({})
    required_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        _require_identifier(self.request_id, "request_id")
        _require_identifier(self.function_id, "function_id")
        _require_mode(self.mode)
        if not isinstance(self.compilation_options, CompilationOptions):
            raise TypeError("compilation_options must be CompilationOptions")
        if self.mode != self.compilation_options.mode:
            raise ExecutionBoundaryError(
                "execution_request_invalid",
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
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionRequest:
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
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreparedExecution:
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
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionResult:
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
    function: Callable[..., Any],
    request: ExecutionRequest,
    backend: ExecutionBackend,
) -> PreparedExecution:
    """Validate policy and create an opaque backend preparation without executing."""

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
                "execution_context_mismatch",
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
    function: Callable[..., Any],
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
    if getattr(backend, "backend_id", None) != context.backend_id:
        raise ExecutionBoundaryError(
            "execution_context_mismatch",
            "execution backend does not match runtime context backend",
            context_backend_id=context.backend_id,
            backend_id=getattr(backend, "backend_id", None),
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
